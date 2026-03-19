import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("tiger.png")
img = img[:,:,0].copy()
bright = img + 50
bright = np.clip(bright, 0, 255) 
rotated = np.rot90(bright, k=-1) 
mirror = np.fliplr(rotated)
reduced = mirror[::10, ::10]
h, w = reduced.shape
result = np.zeros_like(reduced)
result[:, w//4:w//2] = reduced[:, w//4:w//2]
plt.imshow(result, cmap="gray")
plt.show()


