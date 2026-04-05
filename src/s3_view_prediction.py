import cv2
import numpy as np
from s1_sfm import CameraModel, Pose

# Note: the "depth map" here is a depth map for the comparison frame.
def get_view_prediction(img_ref, depth_map, pose_ref, pose_comp, camera_model):
    # Get image size, and matrix for back projection
    h, w = img_ref.shape
    K = camera_model.K
    K_inv = np.linalg.inv(K)

    # Convert pixels coordinates into column form (i.e. create [X1, X2, ...] where Xi are Homogeneous coords)
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    pixels_coords = np.stack([x, y, np.ones_like(x)], axis=-1)
    pixels_coords = pixels_coords.reshape(-1, 3).T

    # Back-project pixels to get the 3D coords in the comparison coordinate form
    # If we assume that camera center is the origin, we just need to do K_inv * pixel_coords
    # which gives normalized 3D coords, which are restored by multiplying by the depth
    cam_coords_comp = K_inv @ pixels_coords

    # Normalize to unit direction before applying Euclidean depth
    norms = np.linalg.norm(cam_coords_comp, axis=0, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    cam_coords_comp = (cam_coords_comp / norms) * depth_map.flatten()

    # Convert comparison camera coordinates to world coordinates
    world_coords = pose_comp.camera_to_world(cam_coords_comp.T)

    # Convert world coordinates into comparison camera frame
    cam_coords_ref = pose_ref.world_to_camera(world_coords).T

    # Project to comparison camera image coordinates
    pixel_coords_ref = K @ cam_coords_ref

    # Avoid division by zero and get coordinates in inhomogeneous coords
    z = pixel_coords_ref[2, :] + 1e-6
    x_ref_flat = (pixel_coords_ref[0, :] / z)
    y_ref_flat = (pixel_coords_ref[1, :] / z)

    # Pixels must land inside the boundaries of the reference frame
    valid_mask = (x_ref_flat >= 0) & (x_ref_flat < w) & (y_ref_flat >= 0) & (y_ref_flat < h)

    # Also verify that the points lie "infront of" the camera
    valid_z = cam_coords_ref[2, :] > 1e-6
    valid_mask = valid_z & valid_mask

    valid_mask = valid_mask.reshape(h, w)

    x_ref = x_ref_flat.reshape(h, w).astype(np.float32)
    y_ref = y_ref_flat.reshape(h, w).astype(np.float32)

    # Obtain image prediction using pixel values
    img_prediction = cv2.remap(img_ref, x_ref, y_ref, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,borderValue=0)

    # Mask out areas where depth was invalid
    img_prediction[depth_map == 0] = 0
    img_prediction[~valid_mask] = 0

    return img_prediction, valid_mask