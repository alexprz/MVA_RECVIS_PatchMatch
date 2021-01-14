"""Implement the inpainting class."""
import numpy as np
from PIL import Image, ImageDraw
from collections.abc import Iterable


class Bbox():
    """Store a bounding box."""

    def __init__(self, x1, y1, x2, y2):
        """Init a bbox.

        Args:
        -----
            x1 : int
            x2 : int
            y1 : int
            y2 : int

        """
        assert isinstance(x1, int)
        assert isinstance(x2, int)
        assert isinstance(y1, int)
        assert isinstance(y2, int)
        assert x2 > x1 >= 0
        assert y2 > y1 >= 0

        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    @property
    def w(self):
        """Get box width."""
        return self.x2 - self.x1

    @property
    def h(self):
        """Get box height."""
        return self.y2 - self.y1

    @property
    def size(self):
        """Get box (width, height)."""
        return self.w, self.h

    @property
    def coords(self):
        """Get box coordinates."""
        return self.x1, self.y1, self.x2, self.y2

    def pad(self, p):
        """Pad the bbox to extend or reduce its size on all sides.

        Args:
        -----
            p : int (can be negative)

        Returns:
        --------
            Bbox object

        """
        return self.__class__(self.x1-p, self.y1-p, self.x2+p, self.y2+p)

    def zeros(self, subshape=None):
        """Get an array of zeros of the shape of the bbox."""
        if subshape is not None:
            if not isinstance(subshape, Iterable):
                subshape = (subshape,)
            return np.zeros((self.h, self.w, *subshape))
        return np.zeros((self.h, self.w))

    def ones(self, subshape=None):
        """Get an array of ones of the shape of the bbox."""
        if subshape is not None:
            if not isinstance(subshape, Iterable):
                subshape = (subshape,)
            return np.ones((self.h, self.w, *subshape))
        return np.ones((self.h, self.w))

    def iterator(self, flip=False):
        """Iterate over the pixels inside the bbox."""
        range1 = list(range(self.x1, self.x2))
        range2 = list(range(self.y1, self.y2))

        if flip:
            range1.reverse()
            range2.reverse()

        for x in range1:
            for y in range2:
                yield y, x

    def __iter__(self):
        """Iterate over the pixels inside the bbox in raster ordering."""
        return self.iterator()

    def is_inside(self, pixels):
        """Tell if given points are inside the bbox or not.

        Args:
        -----
            pixels: np.array of shape (n, 2)

        Returns:
        --------
            np.array of shape (n, 2)

        """
        pixels = np.array(pixels)
        if pixels.ndim == 1:
            pixels = pixels.reshape(1, -1)

        n = pixels.shape[0]
        x1 = self.x1*np.ones(n)
        y1 = self.y1*np.ones(n)
        x2 = self.x2*np.ones(n)
        y2 = self.y2*np.ones(n)

        xy_min = np.stack((y1, x1), axis=1)
        xy_max = np.stack((y2, x2), axis=1)

        return (xy_min <= pixels) & (pixels < xy_max)


class Inpainting():

    def __init__(self, img, patch_radius, alpha, beta):
        """Init.

        Args:
        -----
            img : PIL image
                Original image
            patch_radius : int
            alpha : float
                Search ratio < 1
            beta : float
                Search range

        """
        assert alpha < 1
        assert isinstance(patch_radius, int) and patch_radius >= 0

        self.B = img
        self.array_B = np.array(img)
        self.bbox_B = Bbox(0, 0, img.width, img.height)
        self.bbox_B_t = self.bbox_B.pad(-patch_radius)
        self.pr = patch_radius
        self.alpha = alpha
        self.beta = img.size[0] if beta is None else beta

    def patch_indices(self, flatten=False):
        n = 2*self.pr + 1
        indices = np.indices((n, n)) - self.pr
        indices = indices.transpose(1, 2, 0)
        if flatten:
            indices = indices.reshape(-1, 2)
        return indices

    def fz(self, phi, y, x, bbox_A_t, bbox_A, indices_B, B):
        z = np.array([y, x])
        h = self.patch_indices(flatten=True)

        d = z[None, :] - h  # (p, 2)
        # assert bbox_A_t.is_inside(d).all()
        # print(d)
        # print(phi.shape)
        # print(d[:, 1]-bbox_A_t.x1-1)
        # print(d[:, 0]-bbox_A_t.y1-1)
        p = phi[d[:, 0]-bbox_A_t.y1-1, d[:, 1]-bbox_A_t.x1-1]  # (p, 2)
        # print(np.max(p[:, 0]))
        # print(np.max(p[:, 1]))

        p = p + h

        # print(p.shape)
        # print(h)
        # print(np.max(p[:, 0]))
        # print(np.max(p[:, 1]))
        # print(self.array_B.shape)
        u_t = self.array_B[p[:, 0], p[:, 1], :]
        # print(u_t.shape)
        u_t = u_t.mean(axis=0)

        # print(u_t.shape)
        # print(u_t)
        return u_t

        # exit()

        M = (p[None, :, :] == indices_B[:, None, :]).astype(int)
        # print(np.unique(M, return_counts=True))
        M = M.any(axis=2)
        M = M.sum(axis=1)
        # print(M.shape)
        # print(B.shape)
        # print(np.unique(M, return_counts=True))
        SM = M.sum()
        # print(SM)
        a = np.inner(M, B.T)/SM
        # print(a.shape)
        print(a)
        # exit()

        return a

        # print(p)

    def image_update(self, phi, bbox_A_t, bbox_A, indices_B, B):

        u = bbox_A.zeros(3)

        for i, (y, x) in enumerate(bbox_A):
            # print(i)
            u[y-bbox_A.y1-1, x-bbox_A.x1-1, :] = self.fz(phi, y, x, bbox_A_t, bbox_A, indices_B, B)

        return u

    def map_update(self, u, bbox_A_t, n_iter):
        return self.patch_match(u, bbox_A_t, n_iter)

    def inpaint(self, bbox, n_iter, n_iter_pm):
        """Inpaint the image at the given bounding box.

        Args:
        -----
            bbox : size 4 tuple storing (x1, y1, x2, y2).

        Returns:
        --------
            ?

        """
        bbox_A = Bbox(*bbox)  # bbox of the area to inpaint (img A)
        bbox_A_t = bbox_A.pad(self.pr)  # extended bbox to patch radius

        B_masked = self.B.copy()
        # img_draw = ImageDraw.Draw(B_masked)
        # img_draw.rectangle(bbox, fill='black')
        whole_u = np.array(B_masked)

        W, H = self.bbox_B.size
        is_patch_in_A_t = np.zeros((H, W)).astype(bool)
        x1, y1, x2, y2 = bbox_A_t.coords
        is_patch_in_A_t[y1:y2, x1:x2] = True
        is_patch_in_A_t = is_patch_in_A_t.flatten()

        indices_B = np.indices((H, W)).reshape(2, H*W).T
        indices_B = np.delete(indices_B, is_patch_in_A_t, axis=0)
        # # print(indices_B.shape)

        B = np.array(self.B)

        # # phi = 10*bbox_A_t.ones(2)
        # w_t, h_t = bbox_A_t.size
        # # phi = np.random.randint(0, 200, size=(h_t, w_t, 2))

        # is_patch_on_edge = np.zeros((H, W)).astype(bool)
        # # is_patch_on_edge[self.pr+1:-self.pr-1, self.pr+1:-self.pr-1] = True
        # is_patch_on_edge[:self.pr, :] = True
        # is_patch_on_edge[-self.pr:, :] = True
        # is_patch_on_edge[:, :self.pr] = True
        # is_patch_on_edge[:, -self.pr:] = True
        # # print(np.unique(is_patch_on_edge, return_counts=True))
        # is_patch_on_edge = is_patch_on_edge.flatten()

        # indices_B_t = np.indices((H, W)).reshape(2, H*W).T
        # # print(indices_B_t.shape)
        # indices_B_t = np.delete(indices_B_t, (is_patch_on_edge | is_patch_in_A_t), axis=0)
        # # print(indices_B_t.shape)
        # idx = np.random.choice(np.arange(indices_B_t.shape[0]), size=w_t*h_t)
        # phi = indices_B_t[idx].reshape(h_t, w_t, 2)
        # # print(np.max(phi, axis=1))
        # B = np.array(self.B)
        # B = B[indices_B[:, 0], indices_B[:, 1], :]
        # u = self.image_update(phi, bbox_A_t, bbox_A, indices_B, B)

        # # whole_u.paste(Image.fromarray(np.uint8(u)), (bbox_A.x1, bbox_A.y1))

        # whole_u[bbox_A.y1:bbox_A.y2, bbox_A.x1:bbox_A.x2, :] = u

        for k in range(n_iter):
            phi = self.map_update(whole_u, bbox_A_t, n_iter_pm)

            # print(phi.shape)
            phi_r = phi.reshape(-1, 2)
            # x_min = phi_r.min(axis=0)
            assert Bbox(self.pr, self.pr, W-self.pr, H-self.pr).is_inside(phi_r).all()
            # exit()
            # x_min = np.zeros(self.bbox_B.w)
            # img = Image.fromarray(np.uint8(u))
            # img.show()
            # exit()
            u = self.image_update(phi, bbox_A_t, bbox_A, indices_B, B)
            whole_u[bbox_A.y1:bbox_A.y2, bbox_A.x1:bbox_A.x2, :] = u

        img = Image.fromarray(np.uint8(u))
        return img

    def fill_hole(self, x, y, filling_img):
        """Fill the original image with a filling image at given positions."""
        img = self.B.copy()
        img.paste(filling_img, (x, y))
        return img

    def patch_match(self, u, bbox_A_t, n_iter):
        # Randomly init a map phi
        H, W, _ = u.shape

        is_patch_in_A_t = np.zeros((H, W)).astype(bool)
        x1, y1, x2, y2 = bbox_A_t.coords
        is_patch_in_A_t[y1:y2, x1:x2] = True
        is_patch_in_A_t = is_patch_in_A_t.flatten()

        is_patch_on_edge = np.zeros((H, W)).astype(bool)
        is_patch_on_edge[:self.pr, :] = True
        is_patch_on_edge[-self.pr:, :] = True
        is_patch_on_edge[:, :self.pr] = True
        is_patch_on_edge[:, -self.pr:] = True
        is_patch_on_edge = is_patch_on_edge.flatten()

        indices_B_t = np.indices((H, W)).reshape(2, H*W).T
        indices_B_t = np.delete(indices_B_t, (is_patch_on_edge | is_patch_in_A_t), axis=0)

        w_t, h_t = bbox_A_t.size
        idx = np.random.choice(np.arange(indices_B_t.shape[0]), size=w_t*h_t)

        phi = indices_B_t[idx].reshape(h_t, w_t, 2)

        pr = self.pr

        def D(p1, p2):
            return np.sum(np.power(p1 - p2, 2))

        x0, y0, *_ = bbox_A_t.coords
        print(phi.shape)
        for k in range(1, n_iter+1):
            print(f'Patch match iter {k}')
            flip = (k % 2 == 0)
            for y, x in bbox_A_t.iterator(flip=flip):
                # y = y_A - y0
                # x = x_A - x0
                patch0 = u[y-pr:y+pr, x-pr:x+pr, :]
                # print(y-pr,y+pr, x-pr,x+pr)

                delta = 1 if flip else -1

                y1, x1 = phi[y-y0, x-x0, :]  # middle
                if 0 <= x-x0+delta < phi.shape[1]:
                    y2, x2 = phi[y-y0, x-x0+delta, :]  # left/right
                else:
                    y2, x2 = y, x+delta  #phi[y-y0, x-x0+delta, :]  # left/right

                if 0 <= y-y0+delta < phi.shape[0]:
                    y3, x3 = phi[y-y0+delta, x-x0, :]  # up/down
                else:
                    y3, x3 = y+delta, x #phi[y-y0+delta, x-x0, :]  # up/down

                patch1 = u[y1-pr:y1+pr, x1-pr:x1+pr, :]
                patch2 = u[y2-pr:y2+pr, x2-pr:x2+pr, :]
                patch3 = u[y3-pr:y3+pr, x3-pr:x3+pr, :]

                D1 = D(patch0, patch1)
                D2 = D(patch0, patch2)
                D3 = D(patch0, patch3)

                argmin = y1, x1
                Dmin = D1

                if D2 < Dmin:
                    argmin = y2, x2
                    Dmin = D2

                if D3 < Dmin:
                    argmin = y3, x3
                    Dmin = D3

                y_argmin, x_argmin = argmin
                assert self.pr <= x_argmin < self.bbox_B.w - self.pr
                assert self.pr <= y_argmin < self.bbox_B.h - self.pr
                phi[y-y0, x-x0, :] = argmin

        return phi
