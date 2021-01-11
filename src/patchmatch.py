"""Implement the patch match algorithm."""
import numpy as np


class PatchMatchInpainting():
    """Implement the patch match algorithm for image inpainting."""

    def __init__(self, img, alpha, w=None):
        """Init.

        Args:
        -----
            img : PIL image
                Original image
            alpha : float
                Search ratio
            w : float
                Search range

        """

        self.original_img = img
        self.alpha = alpha
        self.w = img.size[0] if w is None else w

    def inpaint_from_bbox(self, bbox):
        """Inpaint a mask region given by a bbox.

        Args:
        -----
            bbox : size 2 tuple of size 2 tuple

        Returns:
        --------
            masked_img : PIL image
            inpainted_img : PIL image

        """
        pass
