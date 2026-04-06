import numpy as np
import cv2
from skimage.restoration import denoise_tv_chambolle

MAX_WARPS = 5
MAX_OUTER_ITERATIONS = 10
MAX_INNER_ITERATIONS = 30
MAX_LEVEL = 5

def tv_l1(image0, image1):
    I0_pyramid = build_pyramid(image0)
    I1_pyramid = build_pyramid(image1)
    
    # Initialize u = 0, p = 0, and L = max level;
    rows, cols = I1_pyramid[-1].shape
    u1 = np.zeros((rows, cols), dtype=np.float64)
    u2 = np.zeros((rows, cols), dtype=np.float64)
    p1x = np.zeros_like(u1)
    p1y = np.zeros_like(u1)
    p2x = np.zeros_like(u2)
    p2y = np.zeros_like(u2)

    # parameters for theta and lambda (taken from paper may need tuning)
    theta = 0.25
    lambda_ = 30
    time_step = 0.1

    for level in reversed(range(len(I0_pyramid))):
        I0 = I0_pyramid[level]
        I1 = I1_pyramid[level]

        # At the beggining of a new level v is initialized with u
        v1 = np.copy(u1)
        v2 = np.copy(u2)

        # Compute gradients (don't compute on each level)
        Ix, Iy = compute_derivatives(I1)
            
        for warp in range(MAX_WARPS):
            # Re-sample coefficients of ρ using I0 , I1 , and u; (Warping)

            # Warp the target image and gradients
            I1_warped = warp_image(I1, u1, u2)
            Ix_warped = warp_image(Ix, u1, u2)
            Iy_warped = warp_image(Iy, u1, u2)

            # Get temporal gradient
            It = I1_warped - I0

            # Calculate the "squared magnitude" of the gradient of I1_warped
            grad_mag_sq = Ix_warped**2 + Iy_warped**2

            # Help with division by 0 (while still maintaining accuracy)
            grad_mag_sq = np.maximum(grad_mag_sq, 1e-10)

            # Get "u0"
            u0_1 = u1.copy()
            u0_2 = u2.copy() 

            # precompute this part of rho that doesn't depend on current u
            rho_c = I1_warped - I0 - Ix_warped * u0_1 - Iy_warped * u0_2

    
            for _ in range(MAX_OUTER_ITERATIONS):
                # Calculate the linearized residual
                rho = rho_c + Ix_warped * u1 + Iy_warped * u2
                
                # Determine the threshold conditions
                threshold = lambda_ * theta * grad_mag_sq
                
                # Apply thresholding based on the residual (AI used to get the vectorwise updates)
                v1 = np.where(rho < -threshold, u1 + lambda_ * theta * Ix_warped,
                    np.where(rho >  threshold, u1 - lambda_ * theta * Ix_warped,
                                                u1 - rho * Ix_warped / grad_mag_sq))
                                                
                v2 = np.where(rho < -threshold, u2 + lambda_ * theta * Iy_warped,
                    np.where(rho >  threshold, u2 - lambda_ * theta * Iy_warped,
                                                u2 - rho * Iy_warped / grad_mag_sq))

                for _ in range(MAX_INNER_ITERATIONS):
                    # Calculate divergence of p (using backward differences to be adjoint to gradient)
                    div_p1 = backward_div(p1x, p1y)
                            
                    div_p2 = backward_div(p2x, p2y)
                            
                    # update current optic flow
                    u1 = v1 + theta * div_p1
                    u2 = v2 + theta * div_p2

                    # Compute gradients
                    u1x, u1y = forward_grad(u1)
                    u2x, u2y = forward_grad(u2)
                    
                    # Semi-implicit update for p
                    p1x_temp = p1x + (time_step / theta) * u1x
                    p1y_temp = p1y + (time_step / theta) * u1y

                    p2x_temp = p2x + (time_step / theta) * u2x
                    p2y_temp = p2y + (time_step / theta) * u2y
                    
                    # Ensure that magnitude of p <= 1, for now using L2 Norm ***MIGHT NEED TO CHANGE
                    mag_p1 = np.maximum(1.0, np.sqrt(p1x_temp**2 + p1y_temp**2))
                    p1x = p1x_temp / mag_p1
                    p1y = p1y_temp / mag_p1
                    
                    mag_p2 = np.maximum(1.0, np.sqrt(p2x_temp**2 + p2y_temp**2))
                    p2x = p2x_temp / mag_p2
                    p2y = p2y_temp / mag_p2
  
        # Apply Median Filter
        u1 = cv2.medianBlur(u1.astype(np.float32), 5).astype(np.float64)
        u2 = cv2.medianBlur(u2.astype(np.float32), 5).astype(np.float64)

        # Perform prolongation
        if level > 0:
            # Target size is the next finer pyramid level
            target_h, target_w = I0_pyramid[level - 1].shape

            # upscale flow field and crop to match next level
            u1 = prolong_flow(u1)[:target_h, :target_w]
            u2 = prolong_flow(u2)[:target_h, :target_w]

            # upscale dual variable
            zero_border(p1x)
            zero_border(p1y)
            zero_border(p2x)
            zero_border(p2y)

            p1x = prolong_image(p1x)[:target_h, :target_w]
            p1y = prolong_image(p1y)[:target_h, :target_w]
            p2x = prolong_image(p2x)[:target_h, :target_w]
            p2y = prolong_image(p2y)[:target_h, :target_w]

    return u1, u2

def restrict_image(input_image):
    """Downsample using 5x5 binomial filter + subsampling"""

    kernel_1d = np.array([1, 4, 6, 4, 1], dtype=np.float64)
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    kernel_2d /= 256.0

    # Apply smoothing
    smoothed = cv2.filter2D(input_image, -1, kernel_2d, borderType=cv2.BORDER_REFLECT101)

    # Remove odd rows/cols
    return smoothed[::2, ::2]

def prolong_image(input_image):
    """Upsample using zero insertion + 5x5 binomial filter"""

    h, w = input_image.shape
    up = np.zeros((2*h, 2*w), dtype=input_image.dtype)

    # insert zeros into odd columns
    up[::2, ::2] = input_image

    kernel_1d = np.array([1, 4, 6, 4, 1], dtype=np.float64)
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    kernel_2d /= 256.0

    # Apply filter and multiply by 4
    up = cv2.filter2D(up, -1, kernel_2d, borderType=cv2.BORDER_REFLECT101)
    up *= 4.0

    return up

def prolong_flow(u):
    """Upsample flow field (scale vectors by 2)"""
    u_up = prolong_image(u)
    return 2.0 * u_up

def zero_border(p):
    p[0, :] = 0
    p[-1, :] = 0
    p[:, 0] = 0
    p[:, -1] = 0
    return

def forward_grad(f):
    """Forward difference gradient"""
    dx = np.zeros_like(f)
    dy = np.zeros_like(f)
    dx[:, :-1] = f[:, 1:] - f[:, :-1]
    dy[:-1, :] = f[1:, :] - f[:-1, :]
    return dx, dy

def backward_div(px, py):
    """Backward difference divergence (Adjoint to forward_grad)"""
    dx = np.zeros_like(px)
    dy = np.zeros_like(py)
    
    # equations using the "x" direction of p
    # if 1 < i < N, set dx to p_i,j - p_i-1,j
    dx[:, 1:-1] = px[:, 1:-1] - px[:, :-2]

    # if i = 1 set dx to p_i-1,j
    dx[:, 0] = px[:, 0]

    # if i = N, set dx to -p_i-1,j
    dx[:, -1] = -px[:, -2]
    
    # equations using the "y" direction of p
    # if 1 < j < N, set dx to p_i,j - p_i,j-1
    dy[1:-1, :] = py[1:-1, :] - py[:-2, :]
    # if j =1 set dx to p_i,j-1
    dy[0, :] = py[0, :]
    # if j = N, set -p_i,j-1
    dy[-1, :] = -py[-2, :]
    
    return dx + dy

def warp_image(I1, u1, u2):
    """Warps image I1 according to the flow field (u1, u2)."""
    rows, cols = I1.shape
    
    # Create a grid of pixel coordinates
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # Add the flow to the coordinates
    warped_x = (x + u1).astype(np.float32)
    warped_y = (y + u2).astype(np.float32)
    
    # Obtain I1 after applying optic flow (should be ~ the same as I0)
    # This should follow the "bicubic lookup" described in the paper
    I1_warped = cv2.remap(I1, warped_x, warped_y,
                      interpolation=cv2.INTER_CUBIC,
                      borderMode=cv2.BORDER_CONSTANT)

    return I1_warped

def compute_derivatives(image):
    # Higher order central difference computation for gradients
    kernel = np.array([1, -8, 0, 8, -1]) / 12.0
    
    # Compute horizontal derivative
    ix = cv2.filter2D(image, -1, kernel.reshape(1, 5), borderType=cv2.BORDER_REFLECT101)
    
    # Compute vertical derivative
    iy = cv2.filter2D(image, -1, kernel.reshape(5, 1), borderType=cv2.BORDER_REFLECT101)
    
    return ix, iy

def build_pyramid(img):
    pyramid = [img]
    for _ in range(1, MAX_LEVEL):
        if img.shape[0] < 16 or img.shape[1] < 16:
            break
        # Make coarser image
        img = restrict_image(img)

        # Add coarser image to pyramid
        pyramid.append(img)
    return pyramid

# AI USED TO HELP WITH IMPLEMENTATION
def get_forward_backward_mask(flow_fw: np.ndarray, flow_bw: np.ndarray) -> np.ndarray:
    """Computes a mask of valid optical flow vectors using a forward-backward consistency check."""
    h, w = flow_fw.shape[:2]

    # Create a grid of pixel coordinates for Image 1
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # Map Image 1 pixels to their estimated locations in Image 2
    remap_x = (x + flow_fw[..., 0]).astype(np.float32)
    remap_y = (y + flow_fw[..., 1]).astype(np.float32)

    # Use sample backward flow
    sampled_bw_x = cv2.remap(flow_bw[..., 0], remap_x, remap_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    sampled_bw_y = cv2.remap(flow_bw[..., 1], remap_x, remap_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    sampled_bw = np.stack([sampled_bw_x, sampled_bw_y], axis=-1)

    # Calculate the norm of the round-trip error
    round_trip_error = np.linalg.norm(flow_fw + sampled_bw, axis=-1)

    # Get adaptive threshold
    flow_magnitude = np.linalg.norm(flow_fw, axis=-1)
    adaptive_threshold = np.maximum(1.0, 0.05 * flow_magnitude)

    # Create a boolean mask where the error is below the threshold
    valid_mask = round_trip_error < adaptive_threshold

    # Also invalidate pixels that mapped outside the image boundaries
    in_bounds = ((remap_x >= 0) & (remap_x < w) & (remap_y >= 0) & (remap_y < h))
    valid_mask = valid_mask & in_bounds

    return valid_mask

def structure_texture_decomposition_skimage(image, weight=0.2, num_iters=100, alpha=0.95):
    """Extract an images structure & (blended) texture components """
    # Ensure image is float between 0 and 1 for skimage
    img_float = image.astype(np.float32)
    if img_float.max() > 1.0:
        img_float /= 255.0

    # denoise_tv_chambolle returns the smooth "structure"
    # The 'weight' parameter here corresponds to 'theta' in many TV formulations
    structure = denoise_tv_chambolle(img_float, weight=weight, max_num_iter=num_iters)

    # The high-frequency texture is the residual
    texture = img_float - (alpha* structure)

    # Scale back to typical image range if necessary, or keep as float32
    return structure, texture