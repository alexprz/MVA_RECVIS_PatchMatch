"""Implement the patch match algorithm."""
import numpy as np


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

        self.original_img = img
        self.w, self.h = img.size
        self.ps = patch_size
        self.alpha = alpha
        self.beta = img.size[0] if beta is None else beta

    # def _init_offsets(self, img_shape):


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
        # Init offsets
        # f = self._init_offsets()
        img_a = self.original_img.crop(bbox)
        x_a, y_a = bbox[0], bbox[1]
        w_a, h_a = img_a.size

        mask = np.zeros(self.original_img.size)
        mask[x_a:x_a+w_a, y_a:y_a+h_a] = 1

        b_coords = np.where(mask == 0)

        match_idx = np.random.choice(np.arange(b_coords[0].shape[0]), size=w_a*h_a)
        x_pos = b_coords[0][match_idx].reshape(w_a, h_a)
        y_pos = b_coords[1][match_idx].reshape(w_a, h_a)

        positions_in_b = np.stack((x_pos, y_pos), axis=2)
        positions_in_a = np.stack(np.meshgrid(np.arange(w_a), np.arange(h_a)), axis=2)

        # Compute offsets
        f = positions_in_b - positions_in_a



