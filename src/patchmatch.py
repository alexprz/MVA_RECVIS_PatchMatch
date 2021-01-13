"""Implement the patch match algorithm."""
import numpy as np
from PIL import Image, ImageDraw

from sklearn.neighbors import KDTree


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
                Search ratio < 1
            beta : float
                Search range

        """
        assert alpha < 1
        self.img = img
        self.W, self.H = img.size
        self.ps = patch_size
        self.alpha = alpha
        self.beta = img.size[0] if beta is None else beta

        # Compute the size of the patch grid
        self.Wp = self.W//self.ps
        self.Hp = self.H//self.ps

        # Structure the pixels of the original image in patchs
        img_array = np.array(self.img)
        img_array = img_array[:self.Hp*self.ps, :self.Wp*self.ps, :]
        img_array = img_array.reshape(self.Hp, self.ps, self.Wp, self.ps, -1)
        img_array = img_array.swapaxes(1, 2)
        self.img_patches = img_array

    def _pixel_to_patch_coords(self, X, Y):
        """Return patch coordinates from pixel ones.

        Args:
        -----
            X : int
            Y : int

        Returns:
        --------
            Xp : int
            Yp : int

        """
        assert X < self.W
        assert Y < self.H

        Xp = X//self.ps
        Yp = Y//self.ps

        assert Xp < self.Wp
        assert Yp < self.Hp

        return Xp, Yp

    def _pixel_to_patch_bbox(self, bbox):
        """Return patch bbox from pixel one."""
        X1, Y1, X2, Y2 = self._coords_from_bbox(bbox)
        X1p, Y1p = self._pixel_to_patch_coords(X1, Y1)
        X2p, Y2p = self._pixel_to_patch_coords(X2, Y2)
        return X1p, Y1p, X2p, Y2p

    @staticmethod
    def _coords_from_bbox(bbox):
        """Get coordinates from bbox. Make future change in ordering easier."""
        x1, y1, x2, y2 = bbox
        return x1, y1, x2, y2

    @staticmethod
    def _get_bbox_wh(bbox):
        """Compute width and height of the bbox."""
        x1, y1, x2, y2 = PatchMatchInpainting._coords_from_bbox(bbox)
        return x2-x1, y2-y1

    @staticmethod
    def _get_mask(img_shape, bbox):
        """Get a boolean array with 0 outside the bbox and 1 inside."""
        x1, y1, x2, y2 = PatchMatchInpainting._coords_from_bbox(bbox)
        mask = np.zeros(img_shape)
        mask[x1:x2, y1:y2] = 1
        return mask

    @staticmethod
    def _init_offsets(img_shape, bbox):
        """Randomly intialize the deplacement field."""
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

    def _init_field(self, indices_Bp, wp, hp):
        idx = np.random.choice(np.arange(indices_Bp.shape[0]), size=wp*hp)
        field = indices_Bp[idx].reshape(hp, wp, 2)

        return field


    def inpaint_from_bbox(self, bbox, n_iter):
        """Inpaint a masked region given by a bbox.

        Args:
        -----
            bbox : size 4 tuple

        Returns:
        --------
            masked_img : PIL image
            inpainted_img : PIL image

        """
        bboxp = self._pixel_to_patch_bbox(bbox)
        x1p, y1p, x2p, y2p = self._coords_from_bbox(bboxp)
        wp, hp = self._get_bbox_wh(bboxp)
        Wp, Hp = self.Wp, self.Hp

        # Get patch indices for A and B
        # indices_Ap = np.indices((hp, wp)).reshape(2, hp*wp).T
        is_patch_in_A = np.zeros((Hp, Wp)).astype(bool)
        is_patch_in_A[y1p:y2p, x1p:x2p] = True
        is_patch_in_A2D = is_patch_in_A
        is_patch_in_A = is_patch_in_A.flatten()

        indices_Bp = np.indices((Hp, Wp)).reshape(2, Hp*Wp).T
        indices_Bp = np.delete(indices_Bp, is_patch_in_A, axis=0)

        # Retrieve all the patches from B (A is excluded)
        patches_B = self.img_patches[indices_Bp[:, 0], indices_Bp[:, 1], :, :]
        patches_B = patches_B.reshape(indices_Bp.shape[0], -1)

        # Create a kdtree for computing the distance later
        print('Init kdtree')
        kdtree = KDTree(patches_B)

        # Init field
        field = self._init_field(indices_Bp, wp, hp)

        for i, j in self._patch_iterator(field.shape):
            I, J = field[i, j]
            assert not is_patch_in_A2D[I, J]

        return field
        exit()

        # Rescale according to patch size
        # resized_size = np.floor(np.array(self.img.size)/self.ps).astype(int)
        # resized_bbox = np.floor(np.array(bbox)/self.ps).astype(int)

        # Get the two images: A is the masked area and B the rest of the image
        imgA = self.img.crop(bbox)
        imgB = self.get_masked_img(bbox)

        # Precompute the flipped version of A and B.
        # Since the Patch match algorithm requires to change orientation
        # every even iterations, instead of alterning between looking
        # at (left, top) and (right, bottom) neighbors, we always look
        # at (left, top) neighbors but use the flipped version of the image
        # every even iterations instead.
        imgA_flipped = imgA.transpose(Image.FLIP_LEFT_RIGHT)
        imgA_flipped = imgA_flipped.transpose(Image.FLIP_TOP_BOTTOM)

        imgB_flipped = imgB.transpose(Image.FLIP_LEFT_RIGHT)
        imgB_flipped = imgB_flipped.transpose(Image.FLIP_TOP_BOTTOM)

        # Randomly init offsets
        f = self._init_offsets(resized_size, resized_bbox)

        # Distance for offsets
        def D(i1, j1, offset_i, offset_j, flipped):
            """Sum of square distance to the patch behind the mask."""
            i2, j2 = i1 + offset_i, j1 + offset_j
            patch_A_bbox = self._get_patch_bbox(i1, j1)
            patch_B_bbox = self._get_patch_bbox(i2, j2)

            img_a = imgA_flipped if flipped else imgA
            img_b = imgB_flipped if flipped else imgB

            patch_A = np.array(img_a.crop(patch_A_bbox))
            patch_B = np.array(img_b.crop(patch_B_bbox))

            SSD = np.sum(np.power(patch_A - patch_B, 2))

            return SSD

        n_patch_x, n_patch_y, *_ = f.shape
        curr_f = f

        # Iterative update the deplacement field
        for k in range(1, n_iter+1):
            print(f'iter {k}')
            curr_f = np.copy(f)

            # On even iterations we use the flipped images and field
            if k % 2 == 0:
                curr_f = np.flip(curr_f, axis=(0, 1))
                flip = True

            else:
                if k > 2:
                    curr_f = np.flip(curr_f, axis=(0, 1))
                flip = False

            # Iterate over all patches in the masked area
            for i, j in self._patch_iterator(f.shape):
                # Propagation stage
                argmin = curr_f[i, j, :]

                if i > 0:  # if we are in the interior of the area
                    left_offset = curr_f[i-1, j, :]
                else:  # if we are on an edge
                    left_offset = np.array([0, -1])  # not sure I dealt with this correctly

                # If the patch given by the left neighbor is better,
                # we take this deplacement instead
                if left_offset is not None and D(i, j, *left_offset, flip) < D(i, j, *argmin, flip):
                    argmin = left_offset

                if j > 0:  # if we are in the interior of the area
                    up_offset = curr_f[i, j-1, :]
                else:  # if we are on an edge
                    up_offset = np.array([-1, 0])  # not sure I dealt with this correctly

                # If the patch given by the upper neighbor is better,
                # we take this deplacement instead
                if up_offset is not None and D(i, j, *up_offset, flip) < D(i, j, *argmin, flip):
                    argmin = up_offset

                curr_f[i, j, :] = argmin

                # Random search stage
                v0 = curr_f[i, j, :]
                k = 0  # iteration
                argmin = curr_f[i, j, :]
                while True:
                    # Radius of the search
                    radius = self.beta/self.ps*self.alpha**k

                    if radius < 1:
                        # We stop when the radius is too small
                        break

                    # Randomly retrieve a nearby offset
                    eps = np.random.uniform(-1, 1, size=2)
                    u = np.floor(v0 + radius*eps)

                    # Check if it is better
                    if D(i, j, *u, flip) < D(i, j, *argmin, flip):
                        argmin = u

                    k += 1

                curr_f[i, j, :] = argmin

        # If we ended on an even iteration, the field was flipped
        if flip:
            curr_f = np.flip(curr_f, axis=(0, 1))

        return curr_f

    def _patch_iterator(self, a_shape):
        """Iterate over the patches of the masked area."""
        n_patch_x, n_patch_y, *_ = a_shape
        for i in range(n_patch_x):
            for j in range(n_patch_y):
                yield i, j

    def _get_patch_bbox(self, i, j):
        """Get the bbox of a patch given its patch grid coordinates"""
        return i*self.ps, j*self.ps, (i+1)*self.ps, (j+1)*self.ps

    def fill_from_offsets(self, offsets):
        """Get the filled masked area from offsets."""
        n_patch_x, n_patch_y, *_ = offsets.shape

        w, h = n_patch_x*self.ps, n_patch_y*self.ps
        img = Image.new('RGB', (w, h))

        positions_in_a = np.stack(np.meshgrid(np.arange(n_patch_y), np.arange(n_patch_x)), axis=2)

        positions_in_b = np.clip(offsets + positions_in_a, [0, 0], [self.n_patch_x-1, self.n_patch_y-1])

        for i, j in self._patch_iterator((n_patch_x, n_patch_y)):
            patch_bbox = self._get_patch_bbox(*positions_in_b[i, j, :])
            cropped_img = self.img.crop(patch_bbox)
            img.paste(cropped_img, (i*self.ps, j*self.ps))

        return img

    def fill_from_field(self, field):
        """Get the filled masked area from offsets."""
        hp, wp, _ = field.shape

        w, h = wp*self.ps, hp*self.ps
        img = Image.new('RGB', (w, h))

        for yp, xp in self._patch_iterator((hp, wp)):
            Yp, Xp = field[yp, xp]
            patch = self.img_patches[Yp, Xp, :, :, :]
            img_patch = Image.fromarray(patch)
            img.paste(img_patch, (xp*self.ps, yp*self.ps))

        print(field.shape)
        return img

    def get_masked_img(self, bbox):
        """Get the original image with a black mask at the given bbox."""
        x1, y1, x2, y2 = self._coords_from_bbox(bbox)
        img = self.img.copy()
        img_draw = ImageDraw.Draw(img)
        img_draw.rectangle(bbox, fill='black')

        return img

    def fill_hole(self, x, y, filling_img):
        """Fill the original image with a filling image at given positions."""
        img = self.img.copy()
        img.paste(filling_img, (x, y))
        return img



