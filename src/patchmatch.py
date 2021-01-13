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

    def _init_field(self, indices_Bp, wp, hp):
        idx = np.random.choice(np.arange(indices_Bp.shape[0]), size=wp*hp)
        return indices_Bp[idx].reshape(hp, wp, 2)

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
        is_patch_in_A = np.zeros((Hp, Wp)).astype(bool)
        is_patch_in_A[y1p:y2p, x1p:x2p] = True
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

        def D(patch):
            patch = patch.reshape(1, -1)
            return kdtree.query(patch)[0]

        # Iterative update the deplacement field
        for k in range(1, n_iter+1):
            print(f'iter {k}')

            flip = (k % 2 == 0)

            for yp, xp in self._patch_iterator(field.shape, flip):
                argmin_yp, argmin_xp = field[yp, xp, :]
                current_patch = self.img_patches[argmin_yp, argmin_xp, ...]
                current_dist = D(current_patch)

                delta = 1 if flip else -1

                if 0 < xp+delta < field.shape[1]-2:
                    hor_yp, hor_xp = field[yp, xp+delta, :]

                else:
                    hor_yp, hor_xp = yp + y1p, xp+delta + x1p

                hor_patch = self.img_patches[hor_yp, hor_xp, ...]

                # print(D(hor_patch), current_dist)
                if D(hor_patch) < current_dist:
                    argmin_yp, argmin_xp = hor_yp, hor_xp

                if 0 < yp+delta < field.shape[0]-2:
                    vert_yp, vert_xp = field[yp+delta, xp, :]

                else:
                    vert_yp, vert_xp = yp+delta + y1p, xp + x1p

                vert_patch = self.img_patches[vert_yp, vert_xp, ...]

                # print(D(vert_patch), current_patch)
                if D(vert_patch) < current_dist:
                    argmin_yp, argmin_xp = vert_yp, vert_xp

                # print(field[yp, xp], argmin_yp, argmin_xp)
                field[yp, xp] = argmin_yp, argmin_xp

                # Random search stage
                v0 = field[yp, xp, :]
                k = 0  # iteration
                while True:
                    # Radius of the search
                    radius = self.beta/self.ps*self.alpha**k

                    if radius < 1:
                        # We stop when the radius is too small
                        break

                    # Randomly retrieve a nearby offset
                    eps = np.random.uniform(-1, 1, size=2)
                    V0 = v0 + np.array([y1p, x1p])
                    Yp, Xp = np.clip((V0 + radius*eps).astype(int), [0, 0], [self.Hp-1, self.Wp-1])

                    # Check if it is better
                    random_patch = self.img_patches[Yp, Xp, ...]
                    if D(random_patch) < current_dist:
                        field[yp, xp, :] = Xp, Yp

                    k += 1

            mask_filled = self.fill_from_field(field)
            img_filled = self.fill_hole(bbox[0], bbox[1], mask_filled)
            img_filled.show()

        return field

    def _patch_iterator(self, shape, flip=False):
        """Iterate over the patches of the masked area."""
        hp, wp, *_ = shape
        range1 = list(range(hp))
        range2 = list(range(wp))

        if flip:
            range1.reverse()
            range2.reverse()

        for i in range1:
            for j in range2:
                yield i, j

    def _get_patch_bbox(self, i, j):
        """Get the bbox of a patch given its patch grid coordinates"""
        return i*self.ps, j*self.ps, (i+1)*self.ps, (j+1)*self.ps

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



