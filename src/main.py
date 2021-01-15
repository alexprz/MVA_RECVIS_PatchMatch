"""Main script."""
import numpy as np
from patchmatch import PatchMatchInpainting
from inpainting import Inpainting

from PIL import Image


np.random.seed(0)

# Load an image
# img = Image.open('img/lenna.png')
# img = Image.open('img/cow.jpg')
img = Image.open('img/beach.jpg')
# img.show()

# Get the patch match algorithm
# The higher the alpha, the longer the iterations
# pm = PatchMatchInpainting(img, patch_size=5, alpha=0.5, beta=50)
inp = Inpainting(img, patch_radius=2, alpha=0.5, beta=None, sigma=1)

# Choose the bbox of the area to mask
w, h = img.size
print(w, h)
# bbox = (10, 10, 30, 30)
# bbox = (w//2-50, 20, w//2+50, 70)
# bbox = (w//2, 290, w//2+70, 340)
bbox = (897, 428, 1135, 620)
bbox = (424, 203, 533, 292)
# bbox = (300, 90, 310, 100)

mask_filled = inp.inpaint(bbox, n_iter=5, n_iter_pm=5)
img_filled = inp.fill_hole(bbox[0], bbox[1], mask_filled)
masked_img = inp.get_masked_img(bbox)
masked_img.show()
img_filled.show()
exit()

# Get the masked image (for plotting purpose only)
img_mask = pm.get_masked_img(bbox)

# Inpaint the image (f the resulting deplacement field)
field = pm.inpaint_from_bbox(bbox, n_iter=5)

# Retrieve the reconstructed area from the deplacement field
mask_filled = pm.fill_from_field(field)

# Filled the masked image with the reconstruction mask
img_filled = pm.fill_hole(bbox[0], bbox[1], mask_filled)

# Plot all
mask_filled.show()
img_mask.show()
img_filled.show()
