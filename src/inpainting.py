"""Implement the inpainting class."""
import os
import numpy as np
from PIL import Image, ImageDraw
from collections.abc import Iterable
from scipy.ndimage import gaussian_filter
from scipy.stats import multivariate_normal
import argparse

from evaluate import Examiner


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
        return self.x2 - self.x1 + 1

    @property
    def h(self):
        """Get box height."""
        return self.y2 - self.y1 + 1

    @property
    def size(self):
        """Get box (width, height)."""
        return self.w, self.h

    @property
    def coords(self):
        """Get box coordinates."""
        return self.x1, self.y1, self.x2, self.y2

    @classmethod
    def from_mask(cls, mask, keep_true=True):
        mask = np.array(mask).astype(bool)

        if not keep_true:
            mask = ~mask

        idx = np.where(mask)

        h = max(idx[0]) - min(idx[0]) + 1
        w = max(idx[1]) - min(idx[1]) + 1
        x1 = int(idx[1][0])
        y1 = int(idx[0][0])
        x2 = int(x1 + w - 1)
        y2 = int(y1 + h - 1)

        return cls(x1, y1, x2, y2)

    def __getitem__(self, key):
        return self.coords[key]

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
        range1 = list(range(self.x1, self.x2+1))
        range2 = list(range(self.y1, self.y2+1))

        if flip:
            range1.reverse()
            range2.reverse()

        for y in range2:
            for x in range1:
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

        y = pixels[:, 0]
        x = pixels[:, 1]

        return (y >= y1) & (y <= y2) & (x >= x1) & (x <= x2)

        # xy_min = np.stack((y1, x1), axis=1)
        # xy_max = np.stack((y2, x2), axis=1)

        # return (xy_min <= pixels) & (pixels < xy_max)

    def outside(self, y, x):
        if x < self.x1 or x > self.x2 or y < self.y1 or y > self.y2:
            return y, x

        # Both x and y are inside
        # Project x onto boundary
        if abs(x-self.x1) <= abs(x-self.x2):
            xp = self.x1 - 1
        else:
            xp = self.x2 + 1

        # Project y onto boundary
        if abs(y-self.y1) <= abs(y-self.y2):
            yp = self.y1 - 1
        else:
            yp = self.y2 + 1

        return yp, xp


class Inpainting():

    def __init__(self, img, patch_radius, alpha, beta, sigma=2, sigma_img_update=None, sigma_distance=None, init=0, skip_rs=False):
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
        self.bbox_B = Bbox(0, 0, img.width-1, img.height-1)
        self.bbox_B_t = self.bbox_B.pad(-patch_radius)
        self.pr = patch_radius
        self.alpha = alpha
        self.beta = max(img.size[0], img.size[1]) if beta is None else beta
        self.sigma = sigma
        self.init = init
        self.skip_rs = skip_rs

        # self.kernel = self.get_kernel(self.sigma)
        self.kernel_img_update = self.get_kernel(self.sigma)
        self.kernel_distance = self.get_kernel(self.sigma)

        self.valid_patch_idx = self.compute_valid_patch_indices_v2(flip=False)
        self.valid_patch_idx_flip = self.compute_valid_patch_indices_v2(flip=True)

        # print(self.valid_patch_idx)
        # print(self.valid_patch_idx_flip)

        # print(self.kernel[self.valid_patch_idx])
        # exit()


    def patch_indices(self, flatten=False):
        n = 2*self.pr + 1
        indices = np.indices((n, n)) - self.pr
        indices = indices.transpose(1, 2, 0)
        if flatten:
            indices = indices.reshape(-1, 2)
        return indices

    def get_kernel(self, sigma):
        pr = self.pr
        x, y = np.mgrid[-pr:pr+1:1, -pr:pr+1:1]
        pos = np.dstack((x, y))
        g = multivariate_normal(mean=np.zeros(2), cov=sigma*np.eye(2)).pdf(pos)
        g /= np.sum(g)
        return g

    def fz(self, phi, y, x, bbox_A_t):
        z = np.array([y, x])
        h = self.patch_indices(flatten=True)  # (p, 2)

        d = z[None, :] - h  # (p, 2)
        p = phi[d[:, 0]-bbox_A_t.y1-1, d[:, 1]-bbox_A_t.x1-1]  # (p, 2)
        p = p + h  # (p, 2)

        g = self.kernel_img_update.flatten()
        u_t = self.array_B[p[:, 0], p[:, 1], :]  # (p, 3)
        u_t = np.inner(g, u_t.T)

        return u_t

    def image_update(self, phi, bbox_A):
        u = bbox_A.zeros(3)
        bbox_A_t = bbox_A.pad(self.pr)

        for y, x in bbox_A:
            u[y-bbox_A.y1, x-bbox_A.x1, :] = self.fz(phi, y, x, bbox_A_t)

        return u

    def map_update(self, u, bbox_A, n_iter):
        return self.patch_match(u, bbox_A, n_iter)

    def edge_init(self, img_arr, bbox_A):
        img_arr = np.array(img_arr)
        u = bbox_A.zeros(3)
        n, m, _ = u.shape
        M = max(n, m)
        d = abs(m-n)

        x1, y1, x2, y2 = bbox_A.coords
        left = img_arr[y1:y2+1, x1-1]
        right = img_arr[y1:y2+1, x2+1]
        top = img_arr[y1-1, x1:x2+1]
        bot = img_arr[y2+1, x1:x2+1]

        for yy, xx in bbox_A:
            y = yy - y1
            x = xx - x1
            if x <= y <= n-x:
                u[y, x, :] = left[y]
            elif M-x <= y <= x-d:
                u[y, x, :] = right[y]
            elif y < x < m-y and y<n//2:
                u[y, x, :] = top[x]
            elif n-y < x < y+d:
                u[y, x, :] = bot[x]

        return u

    def edge_init_hor(self, img_arr, bbox_A):
        img_arr = np.array(img_arr)
        u = bbox_A.zeros(3)
        n, m, _ = u.shape
        M = max(n, m)
        d = abs(m-n)

        x1, y1, x2, y2 = bbox_A.coords
        left = img_arr[y1:y2+1, x1-1]
        right = img_arr[y1:y2+1, x2+1]

        for yy, xx in bbox_A:
            y = yy - y1
            x = xx - x1
            if x <= m//2:
                u[y, x, :] = left[y]
            else:
                u[y, x, :] = right[y]

        return u

    def edge_init_hor_linear(self, img_arr, bbox_A):
        img_arr = np.array(img_arr)
        u = bbox_A.zeros(3)
        n, m, _ = u.shape
        M = max(n, m)
        d = abs(m-n)

        x1, y1, x2, y2 = bbox_A.coords
        left = img_arr[y1:y2+1, x1-1]
        right = img_arr[y1:y2+1, x2+1]

        u = np.linspace(left, right, m)
        u = np.transpose(u, (1, 0, 2))

        return u


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
        img_draw = ImageDraw.Draw(B_masked)
        img_draw.rectangle(bbox, fill='black')

        if self.init == 0:
            print('Init edge')
            u_init = self.edge_init(B_masked, bbox_A)
        elif self.init == 1:
            print('Init horizontal')
            u_init = self.edge_init_hor(B_masked, bbox_A)
        elif self.init == 2:
            print('Init horizontal linear')
            u_init = self.edge_init_hor_linear(B_masked, bbox_A)

        # u_init = self.edge_init_hor_linear(B_masked, bbox_A)
        u = u_init
        # img = Image.fromarray(np.uint8(u_init))
        # img.show()
        # exit()

        img = self.B.copy()
        draw = ImageDraw.Draw(img)

        draw.rectangle(bbox, fill='black')
        self.draw_rectangle(draw, *bbox_A_t.coords, color=(0, 255, 0))
        self.draw_rectangle(draw, *self.bbox_B_t.coords, color=(0, 255, 0))

        whole_u = np.array(B_masked)
        whole_u[bbox_A.y1:bbox_A.y2+1, bbox_A.x1:bbox_A.x2+1, :] = u_init
        current_img = Image.fromarray(np.uint8(whole_u))
        current_img.show()

        W, H = self.bbox_B.size

        for k in range(1, n_iter+1):
            print(f'inpainting iter {k}')
            phi = self.map_update(whole_u, bbox_A, n_iter_pm)

            phi_r = phi.reshape(-1, 2)
            assert Bbox(self.pr, self.pr, W-self.pr, H-self.pr).is_inside(phi_r).all()

            u = self.image_update(phi, bbox_A)

            whole_u[bbox_A.y1:bbox_A.y2+1, bbox_A.x1:bbox_A.x2+1, :] = u

            current_img = Image.fromarray(np.uint8(whole_u))
            # draw = ImageDraw.Draw(current_img)
            # for i in range(phi.shape[0]):
            #     for j in range(phi.shape[1]):
            #         i2, j2 = phi[i, j]
            #         self.draw_center_patch(draw, j2, i2, (255, 0, 0), r=0)

            current_img.show()

        current_img = Image.fromarray(np.uint8(whole_u))

        # draw = ImageDraw.Draw(current_img)
        # self.draw_rectangle(draw, *bbox_A.coords, color=(0, 255, 0))

        # current_img.show()
        # for i in range(phi.shape[0]):
        #     for j in range(phi.shape[1]):
        #         i2, j2 = phi[i, j]
                # self.draw_center_patch(draw, j2, i2, (255, 0, 0))

        # current_img.show()

        img = Image.fromarray(np.uint8(u))
        return img

    def fill_hole(self, x, y, filling_img):
        """Fill the original image with a filling image at given positions."""
        img = self.B.copy()
        img.paste(filling_img, (x, y))
        return img

    def init_phi(self, H, W, bbox_A_t):
        is_patch_in_A_t = np.zeros((H, W)).astype(bool)
        x1, y1, x2, y2 = bbox_A_t.coords
        is_patch_in_A_t[y1:y2+1, x1:x2+1] = True
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

        return phi

    def draw_center_patch(self, draw, x, y, color, r=None):
        r = self.pr if r is None else r
        draw.line([x-r, y, x+r, y], fill=color)
        draw.line([x, y-r, x, y+r], fill=color)

    def draw_rectangle(self, draw, x1, y1, x2, y2, color):
        draw.line([(x1, y1), (x2, y1)], fill=color)
        draw.line([(x1, y1), (x1, y2)], fill=color)
        draw.line([(x2, y1), (x2, y2)], fill=color)
        draw.line([(x1, y2), (x2, y2)], fill=color)

    def compute_valid_patch_indices(self, flip):
        N = 2*self.pr+1
        patch_mask = np.zeros((N, N)).astype(bool)

        if flip:
            patch_mask[self.pr+1:, :] = True
            patch_mask[self.pr, self.pr+1:] = True

        else:
            patch_mask[:self.pr, :] = True
            patch_mask[self.pr, :self.pr] = True

        return np.where(patch_mask)

    def compute_valid_patch_indices_v2(self, flip):
        N = 2*self.pr+1
        patch_mask = np.zeros((N, N)).astype(bool)

        if flip:
            patch_mask[self.pr:, self.pr+1:] = True
            patch_mask[self.pr+1:, self.pr:] = True

        else:
            patch_mask[:self.pr, :self.pr+1] = True
            patch_mask[:self.pr+1, :self.pr] = True

        return np.where(patch_mask)

    def get_valid_patch_indices(self, flip):
        if flip:
            return self.valid_patch_idx_flip
        return self.valid_patch_idx

    def patch_match(self, u, bbox_A, n_iter):
        # Randomly init a map phi
        H, W, _ = u.shape

        # is_patch_in_A_t = np.zeros((H, W)).astype(bool)
        # x1, y1, x2, y2 = bbox_A_t.coords
        # is_patch_in_A_t[y1:y2, x1:x2] = True
        # is_patch_in_A_t = is_patch_in_A_t.flatten()

        # is_patch_on_edge = np.zeros((H, W)).astype(bool)
        # is_patch_on_edge[:self.pr, :] = True
        # is_patch_on_edge[-self.pr:, :] = True
        # is_patch_on_edge[:, :self.pr] = True
        # is_patch_on_edge[:, -self.pr:] = True
        # is_patch_on_edge = is_patch_on_edge.flatten()

        # indices_B_t = np.indices((H, W)).reshape(2, H*W).T
        # indices_B_t = np.delete(indices_B_t, (is_patch_on_edge | is_patch_in_A_t), axis=0)

        # w_t, h_t = bbox_A_t.size
        # idx = np.random.choice(np.arange(indices_B_t.shape[0]), size=w_t*h_t)

        # phi = indices_B_t[idx].reshape(h_t, w_t, 2)

        bbox_A_t = bbox_A.pad(self.pr)
        phi = self.init_phi(H, W, bbox_A_t)

        current_img = Image.fromarray(np.uint8(u))
        # current_img.show()
        draw = ImageDraw.Draw(current_img)
        # r = self.pr
        for i in range(phi.shape[0]):
            for j in range(phi.shape[1]):
                # print(phi[i, j])
                i2, j2 = phi[i, j]
                # draw.line([(j2-r, i2), (j2+r, i2)], fill=(255, 0, 0))
                # draw.line([(j2, i2-r), (j2, i2+r)], fill=(255, 0, 0))
                # self.draw_center_patch(draw, j2, i2, (255, 0, 0))
                # draw.point(phi[i, j], fill=(255, 0, 0))
        # current_img.show()
        # exit()


        self.draw_rectangle(draw, *bbox_A_t.coords, color=(0, 255, 0))

        pr = self.pr

        def D(p1, p2, flip):
            # idx = self.get_valid_patch_indices(flip)
            # p1 = p1[idx]
            # p2 = p2[idx]
            # d = np.sum(np.power(p1 - p2, 2), axis=2)
            # k = self.kernel_distance
            # return np.inner(d.flatten(), k.flatten())

            idx = self.get_valid_patch_indices(flip)
            p1 = p1[idx]
            p2 = p2[idx]
            d = np.sum(np.power(p1 - p2, 2), axis=1)
            k = self.kernel_distance[idx]
            return np.inner(d, k)

            # exit()
            # d = np.sum(np.power(p1 - p2, 2), axis=2)
            # return np.inner(d.flatten(), self.kernel.flatten())
            # if flip:
            #     d1 = d[pr+1:, :].flatten()
            #     k1 = self.kernel[pr+1:, :].flatten()
            #     d2 = d[pr, pr+1:].flatten()
            #     k2 = self.kernel[pr, pr+1:].flatten()

            # else:
            #     d1 = d[:pr, :].flatten()
            #     k1 = self.kernel[:pr, :].flatten()
            #     d2 = d[pr, :pr].flatten()
            #     k2 = self.kernel[pr, :pr].flatten()

            # return np.inner(k1, d1) + np.inner(k2, d2)

        # def D(p1, p2):
        #     d = np.sum(np.power(p1 - p2, 2), axis=2).flatten()
        #     k = self.kernel.flatten()

        #     return np.inner(k, d)

        x0, y0, *_ = bbox_A_t.coords

        for k in range(1, n_iter+1):
            print(f'Patch match iter {k}')
            flip = (k % 2 == 0)
            # flip = False
            for nb, (y, x) in enumerate(bbox_A_t.iterator(flip=flip)):
                # print(y, x)
                # Propagation stage
                # y = y_A - y0
                # x = x_A - x0
                patch0 = u[y-pr:y+pr+1, x-pr:x+pr+1, :]
                # print(y-pr,y+pr, x-pr,x+pr)

                delta = 1 if flip else -1

                # print(y, x)

                y1, x1 = phi[y-y0, x-x0, :]  # middle
                # self.draw_center_patch(draw, x1, y1, (0, 0, 255))
                y1, x1 = bbox_A_t.outside(y1, x1)
                # self.draw_center_patch(draw, x1, y1, (0, 0, 255))

                if 0 <= x-x0+delta < phi.shape[1]:
                    y2, x2 = phi[y-y0, x-x0+delta, :]  # left/right
                    x2 = np.clip(x2-delta, pr, self.bbox_B.w-pr-1).astype(int)
                    y2, x2 = bbox_A_t.outside(y2, x2)
                else:
                    # y2, x2 = y, x+delta
                    # y2, x2 = None, None
                    y2, x2 = bbox_A_t.outside(y, x+delta)  #phi[y-y0, x-x0+delta, :]  # left/right
                    # self.draw_center_patch(draw, x2, y2, (255, 255, 0))
                    # y2, x2 = y, x+delta  #phi[y-y0, x-x0+delta, :]  # left/right
                    # y2, x2 = None, None#phi[y-y0, x-x0, :]
                    # print("Edge left", y2, x2, y, x)
                # self.draw_center_patch(draw, x2, y2, (255, 0, 0))

                if 0 <= y-y0+delta < phi.shape[0]:
                    y3, x3 = phi[y-y0+delta, x-x0, :]  # up/down
                    y3 = np.clip(y3-delta, pr, self.bbox_B.h-pr-1).astype(int)
                    y3, x3 = bbox_A_t.outside(y3, x3)
                else:
                    # y3, x3 = y+delta, x #phi[y-y0+delta, x-x0, :]  # up/down
                    # y3, x3 = None, None
                    y3, x3 = bbox_A_t.outside(y+delta, x) #phi[y-y0+delta, x-x0, :]  # up/down
                    # self.draw_center_patch(draw, x3, y3, (255, 0, 255))
                    # y3, x3 = None, None #phi[y-y0, x-x0, :]
                    # print("Edge up", y3, x3, y, x)
                # self.draw_center_patch(draw, x3, y3, (0, 255, 0))

                patch1 = u[y1-pr:y1+pr+1, x1-pr:x1+pr+1, :]
                # self.draw_rectangle(draw, x1-pr, y1-pr, x1+pr, y1+pr, color=(0, 255, 0))
                # self.draw_center_patch(draw, x1, y1, color=(0, 255, 0), r=0)
                D1 = D(patch0, patch1, flip)

                if y2 is not None:
                    patch2 = u[y2-pr:y2+pr+1, x2-pr:x2+pr+1, :]
                    # self.draw_rectangle(draw, x2-pr, y2-pr, x2+pr, y2+pr, color=(255, 0, 0))
                    # self.draw_center_patch(draw, x2, y2, color=(255, 0, 0), r=0)
                    # print(y2-pr, y2+pr+1, x2-pr, x2+pr+1)
                    D2 = D(patch0, patch2, flip)
                if y3 is not None:
                    patch3 = u[y3-pr:y3+pr+1, x3-pr:x3+pr+1, :]
                    # self.draw_rectangle(draw, x3-pr, y3-pr, x3+pr, y3+pr, color=(0, 0, 255))
                    # self.draw_center_patch(draw, x3, y3, color=(0, 0, 255), r=0)
                    D3 = D(patch0, patch3, flip)

                argmin = y1, x1
                D_min = D1

                if y2 is not None and D2 < D_min:
                    argmin = y2, x2
                    D_min = D2

                if y3 is not None and D3 < D_min:
                    argmin = y3, x3
                    D_min = D3

                y_argmin, x_argmin = argmin
                assert self.pr <= x_argmin < self.bbox_B.w - self.pr
                assert self.pr <= y_argmin < self.bbox_B.h - self.pr
                phi[y-y0, x-x0, :] = argmin

                # i2, j2 = argmin
                # draw.line([(j2-r, i2), (j2+r, i2)], fill=(255, 255, 0))
                # draw.line([(j2, i2-r), (j2, i2+r)], fill=(255, 255, 0))
                # self.draw_center_patch(draw, j2, i2, (255, 255, 0))

                # break
                # if nb > 10000:
                #     break

                # if (y-y0) % 25 == 0 and x-x0 == 0:
                #     img_arr = self.image_update(phi, bbox_A)
                #     img = Image.fromarray(np.uint8(img_arr))
                #     img.show()

                if self.skip_rs:
                    continue

                # Random search stage
                v0 = phi[y-y0, x-x0, :]
                k = 0
                while self.alpha != 0:
                    # Radius of the search
                    radius = self.beta*self.alpha**k

                    if radius < 1:
                        # We stop when the radius is too small
                        break

                    # Randomly retrieve a nearby offset
                    eps = np.random.uniform(-1, 1, size=2)
                    y_rand, x_rand = (v0 + radius*eps).astype(int)
                    # y_rand, x_rand = np.clip((v0 + radius*eps).astype(int), [pr, pr], [H-pr-1, W-pr-1])

                    if not self.bbox_B_t.is_inside([y_rand, x_rand]).any():
                        k += 1
                        continue

                    if bbox_A_t.is_inside([y_rand, x_rand]).any():
                        k += 1
                        continue

                    self.draw_center_patch(draw, x_rand, y_rand, (255, 0, 255))

                    # Check if it is better
                    patch_rand = u[y_rand-pr:y_rand+pr+1, x_rand-pr:x_rand+pr+1, :]

                    D_rand = D(patch0, patch_rand, flip)
                    if D_rand < D_min:
                        phi[y-y0, x-x0, :] = y_rand, x_rand
                        D_min = D_rand

                    k += 1

                # if (y-y0) % 30 == 0 and x-x0 == 0:
                #     img_arr = self.image_update(phi, bbox_A_t, bbox_A, indices_B, B)
                #     img = Image.fromarray(np.uint8(img_arr))
                #     img.show()

            # img_arr = self.image_update(phi, bbox_A)
            # img = Image.fromarray(np.uint8(img_arr))
            # img.show()

        # self.draw_rectangle(draw, *bbox_A_t.coords, color=(0, 255, 0))
        # current_img.show()

        # for i in range(phi.shape[0]):
        #     for j in range(phi.shape[1]):
        #         i2, j2 = phi[i, j]
        #         self.draw_center_patch(draw, j2, i2, (255, 0, 0))
        # current_img.show()

        return phi

    def get_masked_img(self, bbox):
        """Get the original image with a black mask at the given bbox."""
        x1, y1, x2, y2 = Bbox(*bbox).coords
        img = self.B.copy()
        img_draw = ImageDraw.Draw(img)
        img_draw.rectangle(bbox, fill='black')

        return img

    def get_mask_img(self, bbox):
        """Get a binary image with 1 at the position of the mask."""
        x1, y1, x2, y2 = Bbox(*bbox).coords
        mask_arr = np.zeros_like(self.B).astype(bool)
        mask_arr[y1:y2+1, x1:x2+1] = True
        mask_img = Image.fromarray(np.uint8(mask_arr)*255)
        return mask_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str,
                        help='Folder containing the subfolders to evaluate.')
    parser.add_argument('id', type=int, default=0,
                        help='Id of the image to inpaint.')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Decay ratio of the random search.')
    parser.add_argument('--beta', type=float, default=None,
                        help='Range of the random search.')
    parser.add_argument('--sigma', type=float, default=0.2,
                        help='Width of the gaussian kernel.')
    parser.add_argument('--sigma-update', type=float, default=0.2,
                        help='Width of the gaussian kernel.')
    parser.add_argument('--sigma-dist', type=float, default=0.2,
                        help='Width of the gaussian kernel.')
    parser.add_argument('--pr', type=int, default=2,
                        help='Patch radius (A 0-radius patch is a pixel).')
    parser.add_argument('--init', type=int, default=2,
                        help='Type of initialization')
    parser.add_argument('--skip-rs', nargs='?', type=bool, default=False, const=True,
                        help='Whether to skip the random search.')
    parser.add_argument('--n', type=int, default=2,
                        help='Number of inpainting iterations.')
    parser.add_argument('--n-pm', type=int, default=2,
                        help='Number of PatchMatch iterations.')

    args = parser.parse_args()

    ex = Examiner(root=args.root)

    # for path in ex.img_folder_paths:
    path = ex.img_folder_paths[args.id]
    print(f'Inpainting "{path}"')
    img = Image.open(os.path.join(path, ex.img_filename))
    mask = Image.open(os.path.join(path, ex.mask_filename))

    bbox = Bbox.from_mask(mask, keep_true=False)

    inp = Inpainting(img, patch_radius=args.pr, alpha=args.alpha, beta=args.beta, sigma_distance=args.sigma_dist, sigma_img_update=args.sigma_update, sigma=args.sigma, init=args.init, skip_rs=args.skip_rs)

    inpaint = inp.inpaint(bbox.coords, n_iter=args.n, n_iter_pm=args.n_pm)
    img_inpainted = inp.fill_hole(bbox[0], bbox[1], inpaint)
    img_inpainted.show()
    img_inpainted.save(os.path.join(path, f'inpainted.{ex.ext}'))


