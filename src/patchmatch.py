"""Implement the patch match algorithm."""
import numpy as np
from PIL import Image, ImageDraw


class PatchMatchInpainting():
    """Implement the patch match algorithm for image inpainting."""

    def __init__(self, img, patch_size, alpha, beta=None):
        """Init.

        Args:
        -----
            img : PIL image
                Original image
            patch_size : int
            alpha : float
                Search ratio
            beta : float
                Search range

        """

        self.img = img
        self.w, self.h = img.size
        self.ps = patch_size
        self.alpha = alpha
        self.beta = img.size[0] if beta is None else beta

    @staticmethod
    def _coords_from_bbox(bbox):
        x1, y1, x2, y2 = bbox
        return x1, y1, x2, y2

    @staticmethod
    def _get_bbox_wh(bbox):
        x1, y1, x2, y2 = PatchMatchInpainting._coords_from_bbox(bbox)
        return x2-x1, y2-y1

    @staticmethod
    def _get_mask(img_shape, bbox):
        x1, y1, x2, y2 = PatchMatchInpainting._coords_from_bbox(bbox)
        mask = np.zeros(img_shape)
        mask[x1:x2, y1:y2] = 1
        return mask

    @staticmethod
    def _init_offsets(img_shape, bbox):
        w, h = PatchMatchInpainting._get_bbox_wh(bbox)
        mask = PatchMatchInpainting._get_mask(img_shape, bbox)

        # Retrieve all the coordinates outside the mask (the B image)
        b_coords = np.where(mask == 0)

        # Randomly sample from these coordinates
        match_idx = np.random.choice(np.arange(b_coords[0].shape[0]), size=w*h)
        x_pos = b_coords[0][match_idx].reshape(w, h)
        y_pos = b_coords[1][match_idx].reshape(w, h)
        positions_in_b = np.stack((x_pos, y_pos), axis=2)

        # Coordinates in the mask (image A)
        positions_in_a = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=2)

        # Return offsets
        return positions_in_b - positions_in_a

    def inpaint_from_bbox(self, bbox):
        """Inpaint a mask region given by a bbox.

        Args:
        -----
            bbox : size 4 tuple

        Returns:
        --------
            masked_img : PIL image
            inpainted_img : PIL image

        """
        # Rescale according to patch size
        resized_size = np.floor(np.array(self.img.size)/self.ps).astype(int)
        resized_bbox = np.floor(np.array(bbox)/self.ps).astype(int)

        # Randomly init offsets
        f = self._init_offsets(resized_size, resized_bbox)

        return f

    def _get_patch_bbox(self, i, j):
        return i*self.ps, j*self.ps, (i+1)*self.ps, (j+1)*self.ps

    def fill_from_offsets(self, offsets):
        n_patch_x, n_patch_y, _ = offsets.shape

        w, h = n_patch_x*self.ps, n_patch_y*self.ps
        img = Image.new('RGB', (w, h))

        for i in range(n_patch_x):
            for j in range(n_patch_y):
                patch_bbox = self._get_patch_bbox(*offsets[i, j, :])
                print(patch_bbox)
                cropped_img = self.img.crop(patch_bbox) # offsets[i, j, :])
                print(cropped_img)
                img.paste(cropped_img, (i*self.ps, j*self.ps))

        return img

    def get_masked_img(self, bbox):
        x1, y1, x2, y2 = self._coords_from_bbox(bbox)
        img = self.img.copy()
        img_draw = ImageDraw.Draw(img)
        img_draw.rectangle(bbox, fill='black')

        return img

    def fill_hole(self, x, y, filling_img):
        img = self.img.copy()
        img.paste(filling_img, (x, y))
        return img



