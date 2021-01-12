"""Main script."""
import numpy as np
from patchmatch import PatchMatchInpainting
from PIL import Image


np.random.seed(0)
# img = Image.open('img/lenna.png')
img = Image.open('img/cow.jpg')
img.show()

pm = PatchMatchInpainting(img, patch_size=5, alpha=0.5, beta=50)

# bbox = (10, 10, 30, 30)
w, h = img.size
# bbox = (w//2-50, 20, w//2+50, 70)
# bbox = (w//2, 290, w//2+70, 340)
bbox = (300, 90, 750, 350)

f = pm.inpaint_from_bbox(bbox, n_iter=5)
mask_filled = pm.fill_from_offsets(f)

mask_filled.show()

img_mask = pm.get_masked_img(bbox)
img_mask.show()

img_filled = pm.fill_hole(bbox[0], bbox[1], mask_filled)
img_filled.show()
