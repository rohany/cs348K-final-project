from PIL import Image
import numpy as np
import sys

im = Image.open(sys.argv[1])
ar = np.asarray(im)

out = np.ndarray(ar.shape)
for i in range(0, ar.shape[0]):
    for j in range(0, ar.shape[1]):
        if ar[i, j] <= 40:
            out[i, j] = 0
        else:
            out[i, j] = 255

outIm = Image.fromarray(np.uint8(out))
outIm.save(sys.argv[2])
