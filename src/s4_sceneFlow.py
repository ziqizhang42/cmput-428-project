import numpy as np
import cv2
from s1_sfm import CameraModel, Pose

# Might need to update to work w/ multiple camera models for colmap (depends on how it works)
# Change to align with view prediction

def constrained_scene_flow(depth_ref, pose_ref, camera_model, comp_poses, comp_flows):
    """ Returns Updated Depth Map Given the depth map of the reference image
        the pose of the reference image, the camera model (assumed to be the same for all images)
        a list of poses for the comparison cameras, and a list containing the optic flow for each
        of said images """

    # Get the Neccesary info
    h, w = depth_ref.shape
    K = camera_model.K
    K_inv = np.linalg.inv(K)
    fx = K[0, 0]
    fy = K[1, 1]
    
    # Create array of pixel coordinates
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    pixels_coords = np.stack([x, y, np.ones_like(x)], axis=-1)
    pixels_coords = pixels_coords.reshape(-1, 3).T
    
    # Back-project pixels to rays in camera frame
    rays_cam = (K_inv @ pixels_coords).T

    # Rotate rays to world frame
    rays_world = (pose_ref.R.T @ rays_cam.T).T  # (N, 3)

    # Get ray lines w/ unit vector direction r_j (Equation 4)
    unit_rays = rays_world / np.linalg.norm(rays_world, axis=1, keepdims=True)

    # Get initial 3D world points x_j (Equation 3)
    # depth_ref is Euclidean range (distance from camera center) (use unit rays + center)
    center_ref = pose_ref.camera_center()
    world_points = unit_rays * depth_ref.flatten()[:, None] + center_ref

    # Note: AI used to help with the vectorized computations to avoid 
    # looping as much as possible

    # Accumulate terms for solving the normal equations
    sum_KU = np.zeros(h*w, dtype=np.float32)
    sum_K2 = np.zeros(h*w, dtype=np.float32)
    
    for comp_pose, comp_flow in zip(comp_poses, comp_flows):
        # Transform world points to comparison camera coordinates
        cam_coords_comp = comp_pose.world_to_camera(world_points)
        X = cam_coords_comp[:, 0]
        Y = cam_coords_comp[:, 1]
        Z = cam_coords_comp[:, 2] + 1e-6 # Avoid division by zero
        
        # Project to comparison frame pixel coordinates
        pixel_coords_comp = (K @ cam_coords_comp.T).T
        x_pixel = pixel_coords_comp[:, 0] / Z
        y_pixel = pixel_coords_comp[:, 1] / Z
        
        # Create Mask to only select pixels that are in front of the camera, and actual map onto the image
        valid = (Z > 0.05) & (x_pixel >= 0) & (x_pixel < w) & (y_pixel >= 0) & (y_pixel < h)
        
        # Convert to Meshgrid format to perform mapping
        x_grid = x_pixel.reshape(h, w).astype(np.float32)
        y_grid = y_pixel.reshape(h, w).astype(np.float32)
        
        # Sample the optic flow of each vertex obtained from reference
        vertex_flow = cv2.remap(comp_flow, x_grid, y_grid, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        # Convert so that each row corresponds to the optic flow of a given pixel
        du = vertex_flow.reshape(-1, 2)  # (N, 2)

        # Compute projection Jacobian with respect to camera coords
        J_cam = np.zeros((h*w, 2, 3), dtype=np.float32)
        
        # recall that in pixel coords
        # x = fx*(X/Z) + cx 
        # y = fy*(Y/Z) + cy
        # fx & fy, focal length, cx, cy principal point

        # For Jacobian we have:
        # deriv x w.r.t X, Y & Z
        # likewise w/ y w.r.t X, Y & Z

        # derivs of "x"
        J_cam[:, 0, 0] = fx / Z
        J_cam[:, 0, 1] = 0
        J_cam[:, 0, 2] = -fx * X / (Z**2)

        # derivs of "y"
        J_cam[:, 1, 0] = 0
        J_cam[:, 1, 1] = fy / Z
        J_cam[:, 1, 2] = -fy * Y / (Z**2)
        
        # Get the Jacobian w.r.t the "world coordinates" VIA the chain rule
        J_world = J_cam @ comp_pose.R  # (N, 2, 3)
        
        # Compute K_j^i = J * r_j
        # This is done per pixel i.e. Here each row cooresponds to a jacobian times a ray
        K_ji = np.einsum('nij,nj->ni', J_world, unit_rays)  # (N, 2)
        
        # Add "K*u" to current sum
        sum_KU[valid] += np.sum(K_ji[valid] * du[valid], axis=1)

        # Add K^2 to current sum
        sum_K2[valid] += np.sum(K_ji[valid]**2, axis=1)
        
    # Solve for a vector of lambdas
    valid_lambda = sum_K2 > 1e-6
    lambda_j = np.zeros_like(sum_KU)
    lambda_j[valid_lambda] = sum_KU[valid_lambda] / sum_K2[valid_lambda]

    # Only update pixels that had valid depth
    valid_depth = depth_ref.flatten() > 0
    lambda_j[~valid_depth] = 0.0

    # Perform vectorwise update of rays
    updated_world_points = world_points + unit_rays * lambda_j[:, None]

    # Compute euclidean distance of new points from camera center to get new depth map
    updated_depth = np.linalg.norm(updated_world_points - center_ref, axis=1).reshape(h, w)

    # Zero out originally-invalid pixels
    updated_depth[depth_ref <= 0] = 0.0

    # Apply median filter from Section 2.4.3:
    updated_depth = cv2.medianBlur(updated_depth.astype(np.float32), 3)

    # Error measures (Section 2.4.4, Equation 10)
    # E_s: reprojection residual, average across all comparison frames
    n_comp = len(comp_poses)
    sum_residual = np.zeros(h * w, dtype=np.float32)

    for comp_pose, comp_flow in zip(comp_poses, comp_flows):
        cam_coords_comp = comp_pose.world_to_camera(updated_world_points)
        X = cam_coords_comp[:, 0]
        Y = cam_coords_comp[:, 1]
        Z = cam_coords_comp[:, 2] + 1e-6

        valid = (Z > 0.05)
        x_grid = ((fx * X / Z) + K[0, 2]).reshape(h, w).astype(np.float32)
        y_grid = ((fy * Y / Z) + K[1, 2]).reshape(h, w).astype(np.float32)

        vertex_flow = cv2.remap(comp_flow, x_grid, y_grid, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        du = vertex_flow.reshape(-1, 2)

        J_cam = np.zeros((h * w, 2, 3), dtype=np.float32)
        J_cam[:, 0, 0] = fx / Z
        J_cam[:, 0, 2] = -fx * X / (Z ** 2)
        J_cam[:, 1, 1] = fy / Z
        J_cam[:, 1, 2] = -fy * Y / (Z ** 2)
        J_world = J_cam @ comp_pose.R
        K_ji = np.einsum('nij,nj->ni', J_world, unit_rays)

        residual = K_ji * lambda_j[:, None] - du
        sum_residual += np.linalg.norm(residual, axis=1)

    E_s = (sum_residual / max(n_comp, 1)).reshape(h, w)

    # E_v: visibility measure, approximate surface normals from depth via finite differences in camera frame
    dz_dx = np.zeros_like(updated_depth)
    dz_dy = np.zeros_like(updated_depth)
    dz_dx[:, 1:-1] = (updated_depth[:, 2:] - updated_depth[:, :-2]) / 2.0
    dz_dy[1:-1, :] = (updated_depth[2:, :] - updated_depth[:-2, :]) / 2.0

    # Surface normal in image space
    normals_img = np.stack([-dz_dx, -dz_dy, np.ones_like(updated_depth)], axis=-1)
    n_norm = np.linalg.norm(normals_img, axis=-1, keepdims=True)
    normals_img = normals_img / np.maximum(n_norm, 1e-8)

    # Ray directions in camera frame
    rays_flat = unit_rays.reshape(h, w, 3)
    rays_cam_frame = np.einsum('ij,hwj->hwi', pose_ref.R, rays_flat)
    rays_cam_norm = rays_cam_frame / np.maximum(
        np.linalg.norm(rays_cam_frame, axis=-1, keepdims=True), 1e-8)

    E_v = np.abs(np.sum(rays_cam_norm * normals_img, axis=-1))

    return updated_depth, E_s, E_v
