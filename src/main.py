"""Main script."""
import numpy as np
from patchmatch import PatchMatchInpainting
from PIL import Image


np.random.seed(0)
img = Image.open('img/lenna.png')

pm = PatchMatchInpainting(img, patch_size=5, alpha=0.5)

# bbox = (10, 10, 30, 30)
w, h = img.size
bbox = (w//2-50, h//2-50, w//2+50, h//2+50)

f = pm.inpaint_from_bbox(bbox)
mask_filled = pm.fill_from_offsets(f)

# mask_filled.show()

img_mask = pm.get_masked_img(bbox)
# img_mask.show()

img_filled = pm.fill_hole(bbox[0], bbox[1], mask_filled)
img_filled.show()
