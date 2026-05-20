import dask
import dask.array as da
import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor
import site
import sys
import torch
import torch.nn.functional as F
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops
from pathlib import Path
import os

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional runtime dependency
    ort = None
try:
    import onnx
    from onnx import helper as onnx_helper
    from onnx import TensorProto
except Exception:  # pragma: no cover - optional runtime dependency
    onnx = None
    onnx_helper = None
    TensorProto = None

from model.model import DeepLabV3Plus
from prizm_napari.utils import imadjust, stretchlim


_ONNX_CUDA_DLL_DIRS_CONFIGURED = False
_ONNX_CUDA_DLL_DIRS: list[str] = []
_ONNX_CUDA_DLL_HANDLES = []


def _unique_existing_dirs(paths) -> list[str]:
    seen = set()
    dirs = []
    for path in paths:
        if not path:
            continue
        try:
            resolved = os.path.abspath(os.fspath(path))
        except Exception:
            continue
        if not os.path.isdir(resolved):
            continue
        key = os.path.normcase(resolved)
        if key in seen:
            continue
        seen.add(key)
        dirs.append(resolved)
    return dirs


def _nvidia_site_package_bin_dirs() -> list[str]:
    """
    Locate NVIDIA CUDA/cuDNN runtime DLL folders installed by pip.

    On Windows, `onnxruntime-gpu[cuda,cudnn]` installs DLLs under paths such
    as `<env>/Lib/site-packages/nvidia/cudnn/bin`. ONNX Runtime can preload
    the first-level DLLs, but cuDNN frontend sublibraries may still need the
    folders to be present in the process DLL search path before inference.
    """
    site_roots = []
    try:
        site_roots.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        site_roots.append(site.getusersitepackages())
    except Exception:
        pass
    site_roots.append(str(Path(sys.prefix) / "Lib" / "site-packages"))
    site_roots.extend(p for p in sys.path if "site-packages" in str(p).lower())
    site_roots = _unique_existing_dirs(site_roots)

    components = (
        "cudnn",
        "cublas",
        "cuda_runtime",
        "cufft",
        "curand",
        "cusolver",
        "cusparse",
        "nvjitlink",
        "nvtx",
    )
    candidates = []
    for root in site_roots:
        nvidia_root = Path(root) / "nvidia"
        for component in components:
            candidates.append(nvidia_root / component / "bin")
    return _unique_existing_dirs(candidates)


def _configure_onnx_cuda_dll_search_paths() -> list[str]:
    """
    Add pip-installed NVIDIA CUDA/cuDNN DLL folders to the Windows process.

    This mirrors the manual workaround:
    `%CONDA_PREFIX%\\Lib\\site-packages\\nvidia\\...\\bin` prepended to PATH.
    It is intentionally a no-op on non-Windows platforms.
    """
    global _ONNX_CUDA_DLL_DIRS_CONFIGURED
    global _ONNX_CUDA_DLL_DIRS

    if _ONNX_CUDA_DLL_DIRS_CONFIGURED:
        return list(_ONNX_CUDA_DLL_DIRS)

    _ONNX_CUDA_DLL_DIRS_CONFIGURED = True
    if os.name != "nt":
        return []

    extra = os.environ.get("PRIZM_ONNX_DLL_DIRS", "").strip()
    extra_dirs = [p for p in extra.split(os.pathsep) if p.strip()] if extra else []
    dirs = _unique_existing_dirs(extra_dirs + _nvidia_site_package_bin_dirs())
    if not dirs:
        return []

    existing_path = os.environ.get("PATH", "")
    existing_parts = existing_path.split(os.pathsep) if existing_path else []
    existing_keys = {
        os.path.normcase(os.path.abspath(p))
        for p in existing_parts
        if p
    }
    prepend = [
        d for d in dirs
        if os.path.normcase(os.path.abspath(d)) not in existing_keys
    ]
    if prepend:
        os.environ["PATH"] = os.pathsep.join(prepend + existing_parts)

    add_dll_directory = getattr(os, "add_dll_directory", None)
    if callable(add_dll_directory):
        for dll_dir in dirs:
            try:
                _ONNX_CUDA_DLL_HANDLES.append(add_dll_directory(dll_dir))
            except OSError:
                pass

    _ONNX_CUDA_DLL_DIRS = dirs
    return list(_ONNX_CUDA_DLL_DIRS)


class PRIZMONNXOutOfMemoryError(RuntimeError):
    """Raised when ONNX Runtime hits a GPU memory allocation failure."""


class PRIZMInference:
    def __init__(
        self,
        model_path,
        model_type="auto",
        num_classes=3,
        decoder_atrous_rates=(6, 12, 18),
        backbone="resnet50",
        encoder_depth=5,
        decoder_channels=256,
        encoder_output_stride=16,
        input_channels=1,
        enable_postprocess=False,
        infer_batch_size=1,
        use_amp=True,
    ):
        """
        Initialize the PRIZMInference class.

        Args:
            model_path (str): Path to the pre-trained model file.
            num_classes (int): Number of classes for segmentation.
        """

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.enable_postprocess = bool(enable_postprocess)
        self.infer_batch_size = max(1, int(infer_batch_size))
        self.use_amp = bool(use_amp) and (self.device == "cuda")
        self.input_channels = max(1, int(input_channels))

        self.model_backend = "torch"
        self.onnx_input_name = None
        self.onnx_output_name = None
        self.onnx_fixed_hw = None
        self.onnx_input_scale = 1.0
        self.onnx_label_remap = None
        self.onnx_sessions = []
        self.onnx_session_device_ids = []
        self.onnx_cpu_session = None
        self.onnx_model_path = None
        self.onnx_available_providers = []
        self.onnx_requested_providers = []
        self.onnx_cuda_dll_dirs = []
        self.onnx_cuda_session_errors = []
        self.model = self.load_model(
            model_path=model_path,
            model_type=model_type,
            num_classes=num_classes,
            decoder_atrous_rates=decoder_atrous_rates,
            backbone=backbone,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
            encoder_output_stride=encoder_output_stride,
            input_channels=self.input_channels,
        )

        if self.device == "cuda" and self.model_backend == "torch":
            torch.backends.cudnn.benchmark = True
            self.model = self.model.to(memory_format=torch.channels_last)

    def load_model(
        self,
        model_path,
        model_type="auto",
        num_classes=3,
        decoder_atrous_rates=(6, 12, 18),
        backbone="resnet50",
        encoder_depth=5,
        decoder_channels=256,
        encoder_output_stride=16,
        input_channels=1,
    ):
        """
        Load a pre-trained DeepLabV3+ model from the specified path.

        Args:
            model_path (str): Path to the pre-trained model file.
            num_classes (int): Number of classes for segmentation.

        Returns:
            model (nn.Module): Loaded DeepLabV3+ model.
        """

        model_path = str(model_path)
        requested_type = str(model_type or "auto").strip().lower()
        if requested_type in {"torch", "pt", "pytorch"}:
            requested_type = "pth"
        model_ext = Path(model_path).suffix.lower()
        resolved_type = requested_type
        if resolved_type == "auto":
            resolved_type = "onnx" if model_ext == ".onnx" else "pth"

        if resolved_type == "onnx":
            return self.load_onnx_model(model_path, num_classes=num_classes)
        if resolved_type != "pth":
            raise ValueError(f"Unsupported model_type={model_type!r}; expected one of auto, onnx, pth")

        # Initialize the model
        model = DeepLabV3Plus(
            num_classes=num_classes,
            decoder_atrous_rates=decoder_atrous_rates,
            backbone=backbone,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
            encoder_output_stride=encoder_output_stride,
            in_channels=int(input_channels),
        ).to(self.device)

        # Load the state dictionary from the specified path
        state_dict = torch.load(model_path, map_location=self.device)

        # Update the model's state dictionary with the loaded state dict
        model.load_state_dict(state_dict, strict=False)

        # Set the model to evaluation mode
        model.eval()

        return model

    def load_onnx_model(self, model_path, num_classes=3):
        """
        Load an ONNX segmentation model with ONNX Runtime.

        The ONNX model is expected to accept NCHW float32 input and emit
        per-class logits/probabilities in NCHW layout.
        """
        if ort is None or onnx is None or onnx_helper is None or TensorProto is None:
            raise ImportError(
                "Loading .onnx segmentation models requires `onnx` and `onnxruntime` "
                "(or `onnxruntime-gpu`) to be installed."
            )

        prepared_model_path = self._prepare_onnx_model(model_path)
        self.onnx_model_path = str(prepared_model_path)
        session = None
        self.onnx_sessions = []
        self.onnx_session_device_ids = []
        self.onnx_requested_providers = []
        self.onnx_cuda_session_errors = []
        available = self._get_onnx_available_providers()
        self.onnx_available_providers = sorted(available)

        if "CUDAExecutionProvider" in available:
            candidate_ids = self._candidate_cuda_device_ids()
            if not candidate_ids:
                candidate_ids = [0]
            if not self._onnx_multi_gpu_enabled():
                candidate_ids = candidate_ids[:1]
            for device_id in candidate_ids:
                gpu_session = self._create_onnx_session(prepared_model_path, device_id=device_id)
                if gpu_session is None:
                    continue
                self.onnx_sessions.append(gpu_session)
                self.onnx_session_device_ids.append(device_id)
                if not self._onnx_multi_gpu_enabled():
                    break

        if self.onnx_sessions:
            session = self.onnx_sessions[0]
        else:
            session = self._get_cpu_onnx_session()
            self.onnx_sessions = [session]

        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if not inputs:
            raise RuntimeError(f"ONNX model has no inputs: {model_path}")
        if not outputs:
            raise RuntimeError(f"ONNX model has no outputs: {model_path}")

        self.model_backend = "onnx"
        self.onnx_input_name = inputs[0].name
        self.onnx_output_name = outputs[0].name
        in_shape = list(inputs[0].shape)
        if len(in_shape) >= 4 and isinstance(in_shape[2], int) and isinstance(in_shape[3], int):
            self.onnx_fixed_hw = (int(in_shape[2]), int(in_shape[3]))
        self.onnx_input_scale = self._infer_onnx_input_scale(prepared_model_path)
        self.onnx_label_remap = self._infer_onnx_label_remap(prepared_model_path, int(num_classes))
        return session

    def _get_onnx_available_providers(self) -> set[str]:
        if ort is None:
            return set()

        self.onnx_cuda_dll_dirs = _configure_onnx_cuda_dll_search_paths()
        try:
            preload = getattr(ort, "preload_dlls", None)
            if callable(preload):
                preload(directory="")
        except Exception:
            pass

        try:
            return set(ort.get_available_providers())
        except Exception:
            return set()

    def _candidate_cuda_device_ids(self, min_free_bytes: int = 512 * 1024 * 1024) -> list[int]:
        preferred_device = self._onnx_preferred_device_id()
        if torch.cuda.is_available():
            if preferred_device is not None:
                if 0 <= preferred_device < torch.cuda.device_count():
                    return [int(preferred_device)]
                return []

            ranked = []
            for device_id in range(torch.cuda.device_count()):
                free_bytes = None
                try:
                    free_bytes, _ = torch.cuda.mem_get_info(device_id)
                except Exception:
                    free_bytes = None
                if free_bytes is not None and free_bytes < int(min_free_bytes):
                    continue
                sort_key = free_bytes if free_bytes is not None else -1
                ranked.append((sort_key, device_id))

            if not ranked:
                return list(range(torch.cuda.device_count()))

            ranked.sort(reverse=True)
            return [device_id for _, device_id in ranked]

        if preferred_device is not None and preferred_device >= 0:
            return [int(preferred_device)]

        return [0]

    def _onnx_multi_gpu_enabled(self) -> bool:
        raw = os.environ.get("PRIZM_ONNX_MULTI_GPU", "")
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    def _onnx_preferred_device_id(self) -> int | None:
        raw = os.environ.get("PRIZM_ONNX_DEVICE_ID", "").strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    def _get_cpu_onnx_session(self):
        if self.onnx_model_path is None:
            raise RuntimeError("ONNX model path is not initialized.")
        if self.onnx_cpu_session is None:
            self.onnx_cpu_session = ort.InferenceSession(
                str(self.onnx_model_path),
                providers=["CPUExecutionProvider"],
            )
        return self.onnx_cpu_session

    def _session_device_label(self, session) -> str:
        for idx, known in enumerate(self.onnx_sessions):
            if known is session:
                if idx < len(self.onnx_session_device_ids):
                    return f"GPU {self.onnx_session_device_ids[idx]}"
                break
        if session is self.onnx_cpu_session:
            return "CPU"
        return "unknown device"

    def _format_onnx_memory_error(
        self,
        exc: Exception,
        *,
        batch_size: int | None = None,
        device_id: int | None = None,
    ) -> str:
        text = str(exc).strip()
        first_line = text.splitlines()[0] if text else repr(exc)
        location = f" on GPU {device_id}" if device_id is not None else ""
        batch_text = f" at inference batch size {int(batch_size)}" if batch_size is not None else ""
        return (
            f"PRIZM ran out of GPU VRAM{location}{batch_text} while executing ONNX inference. "
            "Reduce the Batch Segmentation 'Inference Batch Size' and try again. "
            "If needed, close other GPU jobs or use a different GPU.\n\n"
            f"Original ONNX Runtime error: {first_line}"
        )

    def _raise_onnx_memory_error(
        self,
        exc: Exception,
        *,
        batch_size: int | None = None,
        device_id: int | None = None,
    ) -> None:
        raise PRIZMONNXOutOfMemoryError(
            self._format_onnx_memory_error(
                exc,
                batch_size=batch_size,
                device_id=device_id,
            )
        ) from exc

    def _create_onnx_session(self, model_path: str, device_id: int | None = None):
        self.onnx_cuda_dll_dirs = _configure_onnx_cuda_dll_search_paths()
        providers = ["CPUExecutionProvider"]
        available = self._get_onnx_available_providers()
        if "CUDAExecutionProvider" in available and device_id is not None:
            providers = [
                ("CUDAExecutionProvider", {"device_id": int(device_id)}),
                "CPUExecutionProvider",
            ]
        self.onnx_requested_providers.append(list(providers))
        try:
            session = ort.InferenceSession(str(model_path), providers=providers)
        except Exception as exc:
            if device_id is not None:
                self.onnx_cuda_session_errors.append(str(exc).strip() or repr(exc))
            if self._is_onnx_memory_error(exc):
                self._raise_onnx_memory_error(
                    exc,
                    batch_size=self.infer_batch_size,
                    device_id=device_id,
                )
            return None
        if device_id is not None and "CUDAExecutionProvider" not in set(session.get_providers()):
            return None
        return session

    def _run_onnx_session(self, session, batch_np: np.ndarray) -> np.ndarray:
        return session.run(
            [self.onnx_output_name],
            {self.onnx_input_name: batch_np},
        )[0]

    def _run_onnx_batch(self, batch_np: np.ndarray) -> np.ndarray:
        sessions = self.onnx_sessions or [self.model]
        if len(sessions) <= 1 or int(batch_np.shape[0]) <= 1:
            try:
                return self._run_onnx_session(sessions[0], batch_np)
            except Exception as exc:
                if self._is_onnx_memory_error(exc):
                    device_id = self.onnx_session_device_ids[0] if self.onnx_session_device_ids else None
                    self._raise_onnx_memory_error(
                        exc,
                        batch_size=int(batch_np.shape[0]),
                        device_id=device_id,
                    )
                raise

        n_items = int(batch_np.shape[0])
        n_sessions = min(len(sessions), n_items)
        chunk_ranges = []
        start = 0
        for session_idx in range(n_sessions):
            remaining_items = n_items - start
            remaining_sessions = n_sessions - session_idx
            chunk_len = int(math.ceil(remaining_items / remaining_sessions))
            end = start + chunk_len
            chunk_ranges.append((session_idx, start, end))
            start = end

        outputs = [None] * len(chunk_ranges)
        with ThreadPoolExecutor(max_workers=n_sessions) as executor:
            future_map = {
                executor.submit(
                    self._run_onnx_session,
                    sessions[session_idx],
                    batch_np[start:end],
                ): (slot_idx, start, end)
                for slot_idx, (session_idx, start, end) in enumerate(chunk_ranges)
            }
            try:
                for future, (slot_idx, _start, _end) in future_map.items():
                    outputs[slot_idx] = np.asarray(future.result())
            except Exception as exc:
                if self._is_onnx_memory_error(exc):
                    device_id = self.onnx_session_device_ids[0] if self.onnx_session_device_ids else None
                    self._raise_onnx_memory_error(
                        exc,
                        batch_size=int(batch_np.shape[0]),
                        device_id=device_id,
                    )
                raise

        return np.concatenate(outputs, axis=0)

    def _is_onnx_memory_error(self, exc: Exception) -> bool:
        text = str(exc).lower()
        return (
            "failed to allocate memory" in text
            or "cuda out of memory" in text
            or "bfc_arena" in text
            or "cudnn_status_alloc_failed" in text
            or "cublas_status_alloc_failed" in text
            or "cublas failure 3" in text
            or "alloc_failed" in text
        )

    def _prepare_onnx_model(self, model_path: str) -> str:
        """
        Patch MATLAB-exported ONNX crop subgraphs that mix int64 Shape outputs
        with float arithmetic. The patched file is cached next to the source.
        """
        model_path = str(model_path)
        if model_path.endswith(".ortfixed.onnx"):
            return model_path
        patched_path = f"{model_path}.ortfixed.onnx"
        src_mtime = os.path.getmtime(model_path)
        if os.path.exists(patched_path) and os.path.getmtime(patched_path) >= src_mtime:
            return patched_path

        model = onnx.load(model_path)
        graph = model.graph

        # Insert Cast(float) after dec_crop* Shape nodes and route crop arithmetic
        # through the float copies. This preserves the graph semantics while making
        # it valid for ONNX Runtime's stricter type checking.
        shape_outputs_to_cast = {
            out
            for node in graph.node
            if node.name.startswith("dec_crop") and node.op_type == "Shape"
            for out in node.output
        }
        if not shape_outputs_to_cast:
            onnx.save(model, patched_path)
            return patched_path

        new_nodes = []
        cast_map = {}
        for node in graph.node:
            new_nodes.append(node)
            if node.op_type == "Shape" and node.name.startswith("dec_crop"):
                original_out = node.output[0]
                cast_out = f"{original_out}_float"
                cast_map[original_out] = cast_out
                new_nodes.append(
                    onnx_helper.make_node(
                        "Cast",
                        inputs=[original_out],
                        outputs=[cast_out],
                        name=f"{node.name}_CastFloat",
                        to=TensorProto.FLOAT,
                    )
                )

        for node in new_nodes:
            if node.op_type in {"Shape", "Cast"}:
                continue
            for i, inp in enumerate(node.input):
                if inp in cast_map and node.name.startswith("dec_crop"):
                    node.input[i] = cast_map[inp]

        del graph.node[:]
        graph.node.extend(new_nodes)
        onnx.checker.check_model(model)
        onnx.save(model, patched_path)
        return patched_path

    def _infer_onnx_input_scale(self, model_path: str) -> float:
        """
        Heuristic: MATLAB ONNX exports often include an input mean subtraction node.
        If the stored mean is > 1, the graph expects 0..255-scale intensities.
        """
        try:
            model = onnx.load(model_path)
            for init in model.graph.initializer:
                if "Mean" not in init.name:
                    continue
                arr = onnx.numpy_helper.to_array(init)
                finite = arr[np.isfinite(arr)]
                if finite.size and float(np.nanmax(np.abs(finite))) > 1.5:
                    return 255.0
        except Exception:
            pass
        return 1.0

    def _infer_onnx_label_remap(self, model_path: str, num_classes: int):
        """
        MATLAB PRIZM ONNX exports use class order [ventricle, atrium, background].
        Python downstream expects [background, ventricle, atrium].
        """
        try:
            model = onnx.load(model_path)
            producer = str(getattr(model, "producer_name", "") or "")
            if "MATLAB Deep Learning Toolbox Converter for ONNX Model Format" in producer and num_classes == 3:
                # raw ONNX 0,1,2 -> Python 1,2,0
                return np.asarray([1, 2, 0], dtype=np.uint16)
        except Exception:
            pass
        return None

    def preprocess_image(self, image, seg_ch=1) -> torch.Tensor:
        """
        Preprocess the input image for the model.

        Args:
            image (np.ndarray): Input image to be preprocessed.

        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        
        # import pdb; pdb.set_trace()

        seg_ch = int(seg_ch)

        if isinstance(image, da.Array):
            # If the image is a Dask array, then its a multi-frame image
            image = image.compute()
            if image.ndim == 3:
                # Multi-frame grayscale (T,H,W)
                image = np.expand_dims(image, axis=-1)
            elif image.ndim == 4:
                # Multi-frame multi-channel (T,H,W,C)
                if self.input_channels == 1:
                    c = int(image.shape[-1])
                    ch_idx = max(0, min(seg_ch, c - 1))
                    image = image[:, :, :, ch_idx:ch_idx+1]
                else:
                    c = int(image.shape[-1])
                    if c == 1:
                        image = np.repeat(image, self.input_channels, axis=-1)
                    else:
                        # MATLAB-parity path: use selected segmentation channel
                        # and replicate to the model input channel count.
                        ch_idx = max(0, min(seg_ch, c - 1))
                        image = image[:, :, :, ch_idx:ch_idx+1]
                        image = np.repeat(image, self.input_channels, axis=-1)
            # Reshape to (T, C, H, W)
            image = np.transpose(image, (0, 3, 1, 2))
        else:
            # If the image is a numpy array, then its a single-frame image
            if image.ndim == 2:
                # If the image is single-frame grayscale, add channel and batch dimensions
                image = np.expand_dims(image, axis=0)
                if self.input_channels > 1:
                    image = np.repeat(image, self.input_channels, axis=0)
                image = np.expand_dims(image, axis=0)  # Add batch dimension
            elif image.ndim == 3:
                # Shape: (H, W, C)
                if self.input_channels == 1:
                    c = int(image.shape[-1])
                    ch_idx = max(0, min(seg_ch, c - 1))
                    image = image[:, :, ch_idx:ch_idx+1]
                else:
                    c = int(image.shape[-1])
                    if c == 1:
                        image = np.repeat(image, self.input_channels, axis=-1)
                    else:
                        # MATLAB-parity path: use selected segmentation channel
                        # and replicate to the model input channel count.
                        ch_idx = max(0, min(seg_ch, c - 1))
                        image = image[:, :, ch_idx:ch_idx+1]
                        image = np.repeat(image, self.input_channels, axis=-1)
                # Reshape to (C,H,W)
                image = np.transpose(image, (2, 0, 1))
                # Add batch dimension
                image = np.expand_dims(image, axis=0)  # Add batch dimension
            elif image.ndim == 4:
                # Multi-frame multi-channel numpy array: (T, H, W, C)
                if self.input_channels == 1:
                    c = int(image.shape[-1])
                    ch_idx = max(0, min(seg_ch, c - 1))
                    image = image[:, :, :, ch_idx:ch_idx+1]
                else:
                    c = int(image.shape[-1])
                    if c == 1:
                        image = np.repeat(image, self.input_channels, axis=-1)
                    else:
                        ch_idx = max(0, min(seg_ch, c - 1))
                        image = image[:, :, :, ch_idx:ch_idx+1]
                        image = np.repeat(image, self.input_channels, axis=-1)
                image = np.transpose(image, (0, 3, 1, 2))
                
        # image shape = (T, 1, H, W)
        # import pdb; pdb.set_trace()
        
        _, _, h_original, w_original = image.shape

        image_load = []
        for i in range(image.shape[0]):
            # Keep preprocess on CPU; transfer to device in larger inference batches.
            image_torch = torch.from_numpy(image[i].astype(np.float32)).unsqueeze(0)
            
            # import pdb; pdb.set_trace()
            
            # assume image_torch is [C, H, W]
            _, _, h, w = image_torch.shape

            if self.model_backend == "onnx" and self.onnx_fixed_hw is not None:
                target_h, target_w = self.onnx_fixed_hw
                if (h, w) != (target_h, target_w):
                    image_torch = F.interpolate(
                        image_torch,
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    )
            else:
                # Compute smallest multiples of 16 >= h,w.
                # Use pad/crop (not resample) to avoid geometric warping.
                new_h = max(16, math.ceil(h / 16) * 16)
                new_w = max(16, math.ceil(w / 16) * 16)
                pad_h = int(new_h - h)
                pad_w = int(new_w - w)
                if pad_h > 0 or pad_w > 0:
                    # Pad order: (left, right, top, bottom)
                    image_torch = F.pad(image_torch, (0, pad_w, 0, pad_h), mode="replicate")

            # import pdb; pdb.set_trace()

            J = stretchlim(image_torch[0, 0])
            J_modified = J.clone()
            J_modified[0] = J[0] * 1.0
            J_modified[1] = J[1] * 0.90
            if self.model_backend == "onnx":
                image_torch = imadjust(
                    image_torch,
                    in_range=J_modified,
                    out_range=(0.0, float(self.onnx_input_scale)),
                )
            else:
                image_torch = imadjust(image_torch, in_range=J_modified)
            image_load.append(image_torch)
            
            # import pdb; pdb.set_trace()

        image_load = torch.cat(image_load, dim=0)
        
        # import pdb; pdb.set_trace()

        return image_load, h_original, w_original

    def postprocess_image(self, image, seg_mask):
        """
        Postprocess segmentation mask by removing dark connected components.
        
        Args:
            image (torch.Tensor): Original image tensor of shape (1, H, W)
            seg_mask (np.ndarray): Segmentation mask of shape (H, W) where 
                                 0=background, 1=ventricle, 2=atrium
        
        Returns:
            np.ndarray: Processed segmentation mask
        """
        # Convert image tensor to numpy and normalize to [0, 1]
        image_np = image.detach().cpu().numpy()
        if image_np.ndim == 3:
            if image_np.shape[0] == 1:
                image_np = image_np[0]
            else:
                # Use mean intensity map for brightness filtering when model input has >1 channels.
                image_np = np.mean(image_np, axis=0)
        elif image_np.ndim != 2:
            image_np = np.squeeze(image_np)
        
        # Parameters matching MATLAB code
        min_area = 300
        dark_th = 0.15
        p_th = 0.70
        
        # Create binary masks for each class
        ventricular_mask = (seg_mask == 1).astype(bool)
        atrium_mask = (seg_mask == 2).astype(bool)
        
        # Remove small objects (equivalent to bwareaopen)
        if np.any(ventricular_mask):
            ventricular_mask = remove_small_objects(ventricular_mask, min_size=min_area, connectivity=2)
        if np.any(atrium_mask):
            atrium_mask = remove_small_objects(atrium_mask, min_size=min_area, connectivity=2)
        
        # === VENTRICLE brightness filtering ===
        if np.any(ventricular_mask):
            # Label connected components
            ventricle_labels = label(ventricular_mask, connectivity=2)  # 8-connectivity
            ventricle_props = regionprops(ventricle_labels)
            
            if len(ventricle_props) > 1:
                clean_ventricular_mask = np.zeros_like(ventricular_mask, dtype=bool)
                
                for prop in ventricle_props:
                    # Get pixel coordinates for this component
                    coords = prop.coords  # (N, 2) array of (row, col)
                    
                    # Extract pixel values from original image
                    pix_vals = image_np[coords[:, 0], coords[:, 1]]
                    
                    # Calculate ratio of dark pixels
                    dark_ratio = np.mean(pix_vals < dark_th)
                    
                    # Keep component if dark ratio is below threshold (i.e., it's bright enough)
                    if dark_ratio < p_th:
                        clean_ventricular_mask[coords[:, 0], coords[:, 1]] = True
                
                ventricular_mask = clean_ventricular_mask
        
        # === ATRIUM brightness filtering ===
        if np.any(atrium_mask):
            # Label connected components
            atrium_labels = label(atrium_mask, connectivity=2)  # 8-connectivity
            atrium_props = regionprops(atrium_labels)
            
            if len(atrium_props) > 1:
                clean_atrium_mask = np.zeros_like(atrium_mask, dtype=bool)
                
                for prop in atrium_props:
                    # Get pixel coordinates for this component
                    coords = prop.coords  # (N, 2) array of (row, col)
                    
                    # Extract pixel values from original image
                    pix_vals = image_np[coords[:, 0], coords[:, 1]]
                    
                    # Calculate ratio of dark pixels
                    dark_ratio = np.mean(pix_vals < dark_th)
                    
                    # Keep component if dark ratio is below threshold (i.e., it's bright enough)
                    if dark_ratio < p_th:
                        clean_atrium_mask[coords[:, 0], coords[:, 1]] = True
                
                atrium_mask = clean_atrium_mask
        
        # === Apply filtered masks back to segmentation ===
        processed_seg_mask = seg_mask.copy()
        
        # Remove ventricle regions that were filtered out
        processed_seg_mask[(seg_mask == 1) & (~ventricular_mask)] = 0
        
        # Remove atrium regions that were filtered out  
        processed_seg_mask[(seg_mask == 2) & (~atrium_mask)] = 0
        
        return processed_seg_mask

    def infer(self, image, seg_ch=1) -> np.ndarray:
        """
        Perform inference on the input image using the loaded model.

        Args:
            image (np.ndarray): Input image for segmentation.

        Returns:
            np.ndarray: Segmentation output.
        """

        # Preprocess the image
        image_tensor, h_original, w_original = self.preprocess_image(image, seg_ch)
        # image_tensor.shape = (T, 1, H, W)
        masks = []
        
        # import pdb; pdb.set_trace()

        n_frames = int(image_tensor.shape[0])
        start = 0
        current_batch_size = max(1, int(self.infer_batch_size))
        while start < n_frames:
            end = min(start + current_batch_size, n_frames)
            batch = image_tensor[start:end]

            if self.model_backend == "onnx":
                batch_np = batch.detach().cpu().numpy().astype(np.float32, copy=False)
                outputs = self._run_onnx_batch(batch_np)
                outputs = np.asarray(outputs)[..., :h_original, :w_original]
                pred_argmax = np.argmax(outputs, axis=1).astype(np.uint16, copy=False)
                if self.onnx_label_remap is not None:
                    pred_argmax = self.onnx_label_remap[pred_argmax]
            else:
                batch = batch.to(self.device, non_blocking=True)
                if self.device == "cuda":
                    batch = batch.contiguous(memory_format=torch.channels_last)

                with torch.inference_mode():
                    if self.use_amp:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            outputs = self.model(batch)
                    else:
                        outputs = self.model(batch)

                # Crop back to original size after padded inference.
                outputs = outputs[..., :h_original, :w_original]
                pred_argmax = torch.argmax(outputs, dim=1).detach().cpu().numpy()  # (B,H,W)

            if self.enable_postprocess:
                for bi in range(pred_argmax.shape[0]):
                    pred_argmax[bi] = self.postprocess_image(image_tensor[start + bi], pred_argmax[bi])

            masks.append(pred_argmax)
            start = end

        masks = np.concatenate(masks, axis=0).astype(np.uint16)

        return masks
