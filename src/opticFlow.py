import numpy as np

MAX_WARPS = 35
MAX_OUTER_ITERATIONS = 5
MAX_INNER_ITERATIONS = 1


def tv_l1_inner_loop(u1, u2, I_x, I_y, I_t, lambda_=0.25, theta=0.3, time_step=0.25):
    """
    Computes the inner iterations of the TV-L1 optical flow algorithm.
    u1, u2: Current horizontal and vertical flow fields (2D NumPy arrays)
    I_x, I_y: Spatial gradients of the warped image
    I_t: Temporal gradient (I1_warped - I0)
    """
    rows, cols = u1.shape
    
    # Initialize auxiliary variables (v) and dual variables (p)
    v1 = np.copy(u1)
    v2 = np.copy(u2)
    p1x = np.zeros_like(u1)
    p1y = np.zeros_like(u1)
    p2x = np.zeros_like(u2)
    p2y = np.zeros_like(u2)
    
    # Calculate the "squared magnitude" of the gradient of I1_warped
    grad_mag_sq = I_x**2 + I_y**2
    grad_mag_sq[grad_mag_sq == 0] = 1e-10 
    
    for _ in range(MAX_OUTER_ITERATIONS):
        # --- Step 1: Update auxiliary variable v (Point-wise Thresholding) ---
        # Calculate the linearized residual
        rho = I_t + I_x * (u1 - v1) + I_y * (u2 - v2) 
        
        # Determine the threshold conditions
        threshold = lambda_ * theta * grad_mag_sq
        
        # Apply thresholding based on the residual
        v1 = np.where(rho < -threshold, u1 + lambda_ * theta * I_x,
             np.where(rho >  threshold, u1 - lambda_ * theta * I_x,
                                        u1 - rho * I_x / grad_mag_sq))
                                        
        v2 = np.where(rho < -threshold, u2 + lambda_ * theta * I_y,
             np.where(rho >  threshold, u2 - lambda_ * theta * I_y,
                                        u2 - rho * I_y / grad_mag_sq))

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

            p2x_temp = p2x + (time_step / theta) * u2_x
            p2y_temp = p2y + (time_step / theta) * u2_y
            
            # Ensure that magnitude of p <= 1, for now using L2 Norm ***MIGHT NEED TO CHANGE
            mag_p1 = np.maximum(1.0, np.sqrt(p1x_temp**2 + p12_new**2))
            p1x = p1x_temp / mag_p1
            p1y = p1y_temp / mag_p1
            
            mag_p2 = np.maximum(1.0, np.sqrt(p21_new**2 + p22_new**2))
            p21 = p21_new / mag_p2
            p22 = p22_new / mag_p2

    return u1, u2

def restrict_image(input_image):
    pass

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