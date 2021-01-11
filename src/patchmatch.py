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

        self.n_patch_x, self.n_patch_y = np.floor(np.array(img.size)/self.ps).astype(int)

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
        positions_in_a = np.stack(np.meshgrid(np.arange(h), np.arange(w)), axis=2)

        # Return offsets
        return positions_in_b - positions_in_a

    def inpaint_from_bbox(self, bbox, n_iter):
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

        imgA = self.img.crop(bbox)

        imgA_flipped = imgA.transpose(Image.FLIP_LEFT_RIGHT)
        imgA_flipped = imgA_flipped.transpose(Image.FLIP_TOP_BOTTOM)

        imgB_flipped = self.img.transpose(Image.FLIP_LEFT_RIGHT)
        imgB_flipped = imgB_flipped.transpose(Image.FLIP_TOP_BOTTOM)

        # Randomly init offsets
        f = self._init_offsets(resized_size, resized_bbox)

        # Distance for offsets
        def D(i1, j1, offset_i, offset_j, flipped):
            i2, j2 = i1 + offset_i, j1 + offset_j
            patch_A_bbox = self._get_patch_bbox(i1, j1)
            patch_B_bbox = self._get_patch_bbox(i2, j2)

            img_a = imgA_flipped if flipped else imgA
            img_b = imgB_flipped if flipped else self.img

            patch_A = np.array(img_a.crop(patch_A_bbox))
            patch_B = np.array(img_b.crop(patch_B_bbox))

            SSD = np.sum(np.power(patch_A - patch_B, 2))

            return SSD

        curr_f = f

        # Iterative update
        for k in range(1, n_iter+1):
            print(f'iter {k}')
            curr_f = np.copy(f)

            # On even iterations we flip the image
            if k % 2 == 0:
                # curr_f = curr_f.transpose(Image.FLIP_LEFT_RIGHT)
                # curr_f = curr_f.transpose(Image.FLIP_TOP_BOTTOM)
                curr_f = np.flip(curr_f, axis=(0, 1))
                flip = True

            else:
                flip = False

            for i, j in self._patch_iterator(f.shape):
                # Propagation stage
                argmin = curr_f[i, j, :]
                # curr_patch_bbox = ?

                if i > 0:
                    left_offset = curr_f[i-1, j, :]
                else:
                    left_offset = None
                    # left_patch_bbox = self._get_patch_bbox(*left_offset)
                # else:
                #     left_patch_bbox =

                if left_offset is not None and D(i, j, *left_offset, flip) < D(i, j, *argmin, flip):
                    argmin = left_offset

                if j > 0:
                    up_offset = curr_f[i, j-1, :]
                else:
                    up_offset = None
                    # up_patch = self._get_patch_bbox(*up_offset)
                # else:
                #     up_patch =

                if up_offset is not None and D(i, j, *up_offset, flip) < D(i, j, *argmin, flip):
                    argmin = up_offset

                curr_f[i, j, :] = argmin

                # Random search stage

        return curr_f

    def _patch_iterator(self, a_shape):
        n_patch_x, n_patch_y, *_ = a_shape
        for i in range(n_patch_x):
            for j in range(n_patch_y):
                yield i, j

    def _get_patch_bbox(self, i, j):
        return i*self.ps, j*self.ps, (i+1)*self.ps, (j+1)*self.ps

    def fill_from_offsets(self, offsets):
        n_patch_x, n_patch_y, *_ = offsets.shape

        w, h = n_patch_x*self.ps, n_patch_y*self.ps
        img = Image.new('RGB', (w, h))

        positions_in_a = np.stack(np.meshgrid(np.arange(n_patch_y), np.arange(n_patch_x)), axis=2)

        print(self.n_patch_y, self.n_patch_x)
        positions_in_b = np.clip(offsets + positions_in_a, [0, 0], [self.n_patch_x-1, self.n_patch_y-1])

        for i, j in self._patch_iterator((n_patch_x, n_patch_y)):
            patch_bbox = self._get_patch_bbox(*positions_in_b[i, j, :])#offsets[i, j, :])
            cropped_img = self.img.crop(patch_bbox) # offsets[i, j, :])
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



