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
    cam_coords_comp *= depth_map.flatten()

    # Use convert reference camera coordinates to world coordinates
    world_coords = pose_comp.camera_to_world(cam_coords_comp.T)

    # Convert world coordinates into comparison camera frame
    cam_coords_ref = pose_ref.world_to_camera(world_coords).T

    # Project to comparison camera image coordinates
    pixel_coords_ref = K @ cam_coords_ref

    # Avoid division by zero and get coordinates in inhomogeneous coords
    z = pixel_coords_ref[2, :] + 1e-6
    x_ref = (pixel_coords_ref[0, :] / z).reshape(h, w).astype(np.float32)
    y_ref = (pixel_coords_ref[1, :] / z).reshape(h, w).astype(np.float32)

    # Obtain image prediction using pixel values
    img_prediction = cv2.remap(img_ref, x_ref, y_ref, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,borderValue=0)

    return img_prediction