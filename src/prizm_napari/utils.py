import torch
import os
import numpy as np
import cv2
import imageio

def find_actual_major_axis_endpoints(mask):
    """Find the actual mask boundary endpoints along the major axis direction"""
    m = (mask > 0).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    
    # Get the largest contour
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 5:
        return None, None
    
    # Fit ellipse to get major axis direction
    (cx, cy), (axis1, axis2), angle = cv2.fitEllipse(c)
    
    # Ensure major axis is the longer one
    if axis1 >= axis2:
        major_angle = angle
    else:
        major_angle = angle + 90
    
    # Convert angle to radians
    major_angle_rad = np.radians(major_angle)
    major_vector = np.array([np.cos(major_angle_rad), np.sin(major_angle_rad)])
    
    # Convert contour to points
    contour_points = c.reshape(-1, 2)
    
    # Project all contour points onto the major axis direction
    center = np.array([cx, cy])
    projections = []
    
    for point in contour_points:
        # Vector from center to point
        vec = point - center
        # Project onto major axis direction
        projection = np.dot(vec, major_vector)
        projections.append(projection)
    
    # Find the points with maximum and minimum projections
    max_proj_idx = np.argmax(projections)
    min_proj_idx = np.argmin(projections)
    
    # Get the actual boundary points
    pos_endpoint = contour_points[max_proj_idx]
    neg_endpoint = contour_points[min_proj_idx]
    
    return pos_endpoint, neg_endpoint


def find_boundary_points(mask):
    """Find top and bottom points of a binary mask contour."""
    m = (mask > 0).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 3:
        return None, None
    contour_points = c.reshape(-1, 2)
    y_coords = contour_points[:, 1]
    top_idx = int(np.argmin(y_coords))
    bottom_idx = int(np.argmax(y_coords))
    top_point = contour_points[top_idx]
    bottom_point = contour_points[bottom_idx]
    return top_point, bottom_point

def overlay_time_series(image_ts, mask_ts, stats_df, viz_data=None, output_dir=None, mask_alpha=0.2):
    """
    Overlay segmentation masks and anatomical features on a grayscale time series with semi-transparent masks,
    computing ventricle axes from the mask itself and V-A relationships from analysis data.

    Args:
        image_ts : dask.array or ndarray, shape (T, H, W)
            Grayscale image time series.
        mask_ts : dask.array or ndarray, shape (T, H, W)
            Segmentation masks with labels: 0=background, 1=ventricle, 2=atrium.
        stats_df : pandas.DataFrame, length T
            DataFrame with per-frame measurements. Must contain at least:
            ['AtriumCentroid'] (ventricle centroid / axes optional).
        viz_data : dict or None
            Visualization data from analysis including V-A relationships, major/minor axes, etc.
        output_dir : str or None
            Optional directory path to save each overlaid frame as PNG.
        mask_alpha : float
            Alpha transparency for mask fill (0.0-1.0).

    Returns:
        overlay_ts : ndarray, shape (T, H, W, 3), dtype=uint8
            RGB image time series with overlays.
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    T, H, W = image_ts.shape
    overlay_ts = np.zeros((T, H, W, 3), dtype=np.uint8)

    # Drawing parameters
    line_width = 2  # Changed to integer for OpenCV compatibility
    centroid_line_width = 1
    circle_radius = 12

    # Colors as RGB tuples
    vent_color = (0, 0, 255)           # Blue for ventricle
    atr_color  = (0, 255, 0)           # Green for atrium
    cent_line_color = (0, 0, 0)        # Black for line between centroids
    vent_major_color = (255, 255, 0)   # Yellow for ventricle major axis (like MATLAB)
    vent_minor_color = (255, 0, 0)     # Red for ventricle minor axis (like MATLAB)
    va_center_color = (0, 255, 255)    # Cyan for V-A center line
    va_top_color = (0, 255, 255)       # Cyan for V-A top line
    va_bottom_color = (0, 255, 255)    # Cyan for V-A bottom line

    for i in range(T):
        # Load and normalize frame
        frame = image_ts[i]
        mask  = mask_ts[i]
        if hasattr(frame, 'compute'):
            frame = frame.compute()
        if hasattr(mask, 'compute'):
            mask = mask.compute()

        gray = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        # Semi-transparent mask overlay
        overlay = np.zeros_like(img_rgb)
        vent_pixels = (mask == 1)
        atr_pixels  = (mask == 2)
        overlay[vent_pixels] = vent_color
        overlay[atr_pixels]  = atr_color
        cv2.addWeighted(overlay, mask_alpha, img_rgb, 1 - mask_alpha, 0, img_rgb)

        # Draw outlines
        for label, color in [(1, vent_color), (2, atr_color)]:
            lbl_mask = (mask == label).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(lbl_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_rgb, cnts, -1, color, line_width)

        # ---- VENTRICLE: compute ellipse from mask ----
        vent_mask = (mask == 1).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(vent_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ellipse_center = None
        major_len = minor_len = angle_deg = None

        if cnts:
            # pick the largest contour
            largest = max(cnts, key=cv2.contourArea)
            if len(largest) >= 5:
                ellipse = cv2.fitEllipse(largest)
                (cx, cy), (axis1, axis2), ang = ellipse

                # ensure axis1 ≥ axis2
                if axis1 >= axis2:
                    major_len, minor_len = axis1/2.0, axis2/2.0
                    angle_deg = ang
                else:
                    major_len, minor_len = axis2/2.0, axis1/2.0
                    angle_deg = ang + 90.0

                ellipse_center = (int(cx), int(cy))

        # fallback to stats_df if ellipse couldn't be fit
        row = stats_df.iloc[i]
        if ellipse_center is None:
            vent_centroid = row['VentricularCentroid']
            vx = int(vent_centroid[0])  # x coordinate
            vy = int(vent_centroid[1])  # y coordinate
            # Use MATLAB approach for fallback
            vent_pos, vent_neg = find_actual_major_axis_endpoints(vent_mask)
            if vent_pos is not None and vent_neg is not None:
                # Calculate major angle from endpoints
                major_vector = vent_pos - vent_neg
                angle_deg = np.degrees(np.arctan2(major_vector[1], major_vector[0]))
                major_len = np.linalg.norm(major_vector) / 2.0
                minor_len = row['minorAxis_center'] / 2.0
            else:
                angle_deg = row['VA_Angle_Center']  # Use V-A angle as reference
                major_len = row['majorAxisLength'] / 2.0
                minor_len = row['minorAxis_center'] / 2.0
            ellipse_center = (vx, vy)

        vx, vy = ellipse_center  # ventricle centroid for drawing

        # Get actual ventricle major axis endpoints from mask boundary
        vent_pos, vent_neg = find_actual_major_axis_endpoints(vent_mask)
        
        if vent_pos is not None and vent_neg is not None:
            # Use actual mask boundary endpoints for major axis
            p1 = (int(vent_neg[0]), int(vent_neg[1]))
            p2 = (int(vent_pos[0]), int(vent_pos[1]))
            cv2.line(img_rgb, p1, p2, vent_major_color, line_width)
            
            # Calculate major angle from endpoints for minor axis
            major_vector = vent_pos - vent_neg
            angle_deg = np.degrees(np.arctan2(major_vector[1], major_vector[0]))
        else:
            # Fallback to fitted ellipse if mask endpoints not available
            theta = np.deg2rad(angle_deg)
            dx_mj = np.cos(theta) * major_len
            dy_mj = np.sin(theta) * major_len
            p1 = (int(vx - dx_mj), int(vy - dy_mj))
            p2 = (int(vx + dx_mj), int(vy + dy_mj))
            cv2.line(img_rgb, p1, p2, vent_major_color, line_width)

        # Draw three minor axes (red) - exactly like MATLAB code
        if vent_pos is not None and vent_neg is not None:
            # Calculate major axis vector and length
            major_vector = vent_pos - vent_neg
            major_axis_length = np.linalg.norm(major_vector)
            major_angle = np.degrees(np.arctan2(major_vector[1], major_vector[0]))
            major_angle = major_angle % 360
            
            # Calculate minor axis direction (perpendicular to major axis)
            minor_angle = (major_angle + 90) % 360
            minor_angle_rad = np.radians(minor_angle)
            minor_direction = np.array([np.cos(minor_angle_rad), np.sin(minor_angle_rad)])
            
            # Normalize minor direction vector
            minor_direction = minor_direction / np.linalg.norm(minor_direction)
            
            # Calculate offset for upper and lower points (10% of major axis length)
            offset_major = 0.1 * major_axis_length
            
            # Create three centers along major axis
            centers = [
                np.array([vx, vy]),                                    # Center point
                np.array([vx, vy]) + offset_major * major_vector / np.linalg.norm(major_vector),  # Upper shift
                np.array([vx, vy]) - offset_major * major_vector / np.linalg.norm(major_vector)   # Lower shift
            ]
            
            # Draw minor axis at each center
            vent_mask_contour = cnts[0] if cnts else None
            if vent_mask_contour is not None:
                for center in centers:
                    # Walk in positive direction until we hit boundary
                    pos_minor = center.copy()
                    while cv2.pointPolygonTest(vent_mask_contour, (int(pos_minor[0]), int(pos_minor[1])), False) >= 0:
                        last_pos = pos_minor.copy()  # Save last point inside boundary
                        pos_minor = pos_minor + minor_direction
                    
                    # Walk in negative direction until we hit boundary
                    neg_minor = center.copy()
                    while cv2.pointPolygonTest(vent_mask_contour, (int(neg_minor[0]), int(neg_minor[1])), False) >= 0:
                        last_neg = neg_minor.copy()  # Save last point inside boundary
                        neg_minor = neg_minor - minor_direction
                    
                    # Draw minor axis line
                    p3 = (int(last_neg[0]), int(last_neg[1]))
                    p4 = (int(last_pos[0]), int(last_pos[1]))
                    cv2.line(img_rgb, p3, p4, vent_minor_color, line_width)
        else:
            # Fallback to single minor axis if major axis not available
            if vent_pos is not None and vent_neg is not None:
                # Calculate minor axis using perpendicular direction
                minor_angle = (angle_deg + 90) % 360
                minor_angle_rad = np.radians(minor_angle)
                minor_direction = np.array([np.cos(minor_angle_rad), np.sin(minor_angle_rad)])
                
                # Walk from centroid in both directions for minor axis
                pos_pt = np.array([vx, vy])
                neg_pt = np.array([vx, vy])
                
                # Find minor axis endpoints
                vent_mask_contour = cnts[0] if cnts else None
                if vent_mask_contour is not None:
                    while cv2.pointPolygonTest(vent_mask_contour, (int(pos_pt[0]), int(pos_pt[1])), False) >= 0:
                        pos_pt = pos_pt + minor_direction
                    while cv2.pointPolygonTest(vent_mask_contour, (int(neg_pt[0]), int(neg_pt[1])), False) >= 0:
                        neg_pt = neg_pt - minor_direction
                    
                    p3 = (int(neg_pt[0]), int(neg_pt[1]))
                    p4 = (int(pos_pt[0]), int(pos_pt[1]))
                    cv2.line(img_rgb, p3, p4, vent_minor_color, line_width)
            else:
                # Final fallback to fitted ellipse for minor axis
                theta = np.deg2rad(angle_deg)
                dx_mn = -np.sin(theta) * minor_len
                dy_mn =  np.cos(theta) * minor_len
                p3 = (int(vx - dx_mn), int(vy - dy_mn))
                p4 = (int(vx + dx_mn), int(vy + dy_mn))
                cv2.line(img_rgb, p3, p4, vent_minor_color, line_width)

        # ---- ATRIUM centroid and connecting line ----
        atr_centroid = row['AtriumCentroid']
        ax, ay = int(atr_centroid[0]), int(atr_centroid[1])  # x, y coordinates
        cv2.line(img_rgb, (vx, vy), (ax, ay), cent_line_color, line_width)

        # draw centroids
        cv2.circle(img_rgb, (vx, vy), circle_radius, vent_color, centroid_line_width)
        cv2.circle(img_rgb, (ax, ay), circle_radius, atr_color, centroid_line_width)

        # ---- V-A RELATIONSHIP LINES (if viz_data available) ----
        if viz_data is not None:
            # Get V-A relationship data from analysis
            vent_pos, vent_neg = find_actual_major_axis_endpoints(vent_mask)
            atr_top, atr_bottom = find_boundary_points((mask == 2).astype(np.uint8) * 255)
            
            if vent_pos is not None and vent_neg is not None and atr_top is not None and atr_bottom is not None:
                # Determine ventricle top/bottom based on Y-coordinate (MATLAB logic)
                if vent_pos[1] < vent_neg[1]:
                    vent_top = vent_pos
                    vent_bottom = vent_neg
                else:
                    vent_top = vent_neg
                    vent_bottom = vent_pos
                
                # Draw V-A center line (magenta)
                cv2.line(img_rgb, (vx, vy), (ax, ay), va_center_color, line_width)
                
                # Draw V-A top line (cyan)
                cv2.line(img_rgb, 
                        (int(vent_top[0]), int(vent_top[1])), 
                        (int(atr_top[0]), int(atr_top[1])), 
                        va_top_color, line_width)
                
                # Draw V-A bottom line (orange)
                cv2.line(img_rgb, 
                        (int(vent_bottom[0]), int(vent_bottom[1])), 
                        (int(atr_bottom[0]), int(atr_bottom[1])), 
                        va_bottom_color, line_width)

        # Store and save
        overlay_ts[i] = img_rgb
        if output_dir:
            fname = os.path.join(output_dir, f"frame_{i:03d}.png")
            imageio.imwrite(fname, img_rgb)

    return overlay_ts

def stretchlim(image, tol=(0.01, 0.99)):
    """
    Compute the stretch limits for contrast adjustment.
    """
    flattened = image.flatten()
    # Quantile-based estimate is substantially faster than full sort for per-frame calls.
    q = torch.tensor([float(tol[0]), float(tol[1])], device=flattened.device, dtype=flattened.dtype)
    try:
        low, high = torch.quantile(flattened, q)
    except Exception:
        # Fallback for older torch builds: keep previous exact-sort behavior.
        sorted_vals, _ = torch.sort(flattened)
        n_pixels = sorted_vals.shape[0]
        low_idx = int(tol[0] * n_pixels)
        high_idx = int(tol[1] * n_pixels) - 1
        low = sorted_vals[low_idx]
        high = sorted_vals[high_idx]

    # If the image is flat, avoid invalid range
    if low == high:
        return torch.tensor([0.0, 1.0], device=image.device)

    return torch.tensor([low, high], device=image.device)


def imadjust(image, in_range=None, out_range=(0, 1)):
    """
    Adjust the intensity values of an image.
    """
    if in_range is None:
        in_range = (image.min(), image.max())

    # Clip image values to in_range
    image = torch.clamp(image, min=in_range[0], max=in_range[1])

    # Normalize to [0, 1] and scale to out_range
    normalized = (image - in_range[0]) / (in_range[1] - in_range[0])
    scaled = normalized * (out_range[1] - out_range[0]) + out_range[0]

    return scaled
