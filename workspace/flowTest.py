import sys
sys.path.append("../src")

from opticFlow import tv_l1
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load image
img = cv2.imread("../data/rgbd_dataset_freiburg1_desk/rgb/1305031452.791720.png", cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0

dx, dy = 5, 3
M = np.float32([[1, 0, dx],
                [0, 1, dy]])

shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

# Your implementation
u1, u2 = tv_l1(img, shifted)

# OpenCV implementation
tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
flow = tvl1.calc(img.astype(np.float32), shifted.astype(np.float32), None)

u1_cv = flow[..., 0]
u2_cv = flow[..., 1]

# Compare
print("Expected:", dx, dy)
print("Recovered:", np.mean(u1), np.mean(u2))

error = np.sqrt((u1 - u1_cv)**2 + (u2 - u2_cv)**2)
margin = 10
print("Mean error:", np.mean(error[margin:-margin, margin:-margin]))

step = 10  # try 5–15

h, w = u1.shape
x, y = np.meshgrid(np.arange(w), np.arange(h))

plt.quiver(x[::step, ::step],
           y[::step, ::step],
           u1[::step, ::step],
           u2[::step, ::step])
plt.gca().invert_yaxis()
plt.title("Flow field")
plt.show()