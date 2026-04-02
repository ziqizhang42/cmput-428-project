import numpy as np
import cv2

def denoise_depth_map_tvl1(D, I_ref, alpha=10.0, beta=1.0, lambda_data=1.0, num_iters=100):
    """ Minimizes the g-weighted TV-L1 """

    # Compute spatial gradients
    I_ref_f32 = I_ref.astype(np.float32)
    Ix = cv2.Sobel(I_ref_f32, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(I_ref_f32, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(Ix**2 + Iy**2)
    
    # Want Gradient Magitude Between 0 & 1 (so that weight works correctly)
    grad_mag_norm = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-8)
    
    # The isotropic regularisation weight g
    g = np.exp(-alpha * (grad_mag_norm ** beta))
    
    # Initialize variables
    D_prime = D.copy() # New depth Map that's being computed
    D_bar = D_prime.copy() # Over-relaxed variable
    
    px = np.zeros_like(D)
    py = np.zeros_like(D)
    
    # Chambolle-Pock parameters (must satisfy tau * sigma * L^2 <= 1, where L^2 = 8 for 2D gradients)
    tau = 0.35
    sigma = 0.35
    theta = 1.0 # Relaxation parameter
    
    for _ in range(num_iters):
        # Compute gradient w/ forward diff (is relevant for divergence later)
        Dx = np.zeros_like(D_bar)
        Dy = np.zeros_like(D_bar)
        Dx[:, :-1] = D_bar[:, 1:] - D_bar[:, :-1]
        Dy[:-1, :] = D_bar[1:, :] - D_bar[:-1, :]
        
        # Update dual variables
        px_new = px + sigma * Dx
        py_new = py + sigma * Dy
        
        # Ensure that norm of p is <= g
        norm = np.maximum(1.0, np.sqrt(px_new**2 + py_new**2) / np.maximum(g, 1e-8))
        px = px_new / norm
        py = py_new / norm
        
        # Backward differences for the divergence of p (need it to be negative adjoint of gradient)
        div_p = np.zeros_like(px)
        div_p[:, 1:-1] = px[:, 1:-1] - px[:, :-2]
        div_p[:, 0] = px[:, 0]
        div_p[:, -1] = -px[:, -2]
        
        div_p[1:-1, :] += py[1:-1, :] - py[:-2, :]
        div_p[0, :] += py[0, :]
        div_p[-1, :] += -py[-2, :]
        
        D_old = D_prime.copy()
        
        # Proximal gradient step for the L1 data term
        v = D_prime + tau * div_p
        
        # Resolvent operator (soft-thresholding) avoids the instability of np.sign
        # lambda_data controls the trade-off between the data term and smoothing
        D_prime = D + np.clip(v - D, -tau * lambda_data, tau * lambda_data)
        
        # Over-relaxation
        D_bar = D_prime + theta * (D_prime - D_old)
        
    return D_prime