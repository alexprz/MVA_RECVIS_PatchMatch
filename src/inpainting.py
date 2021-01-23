"""Implement the inpainting class."""
import numpy as np
from PIL import Image, ImageDraw
from collections.abc import Iterable
from scipy.ndimage import gaussian_filter
from scipy.stats import multivariate_normal


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

        xy_min = np.stack((y1, x1), axis=1)
        xy_max = np.stack((y2, x2), axis=1)

        return (xy_min <= pixels) & (pixels < xy_max)

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

    def __init__(self, img, patch_radius, alpha, beta, sigma=2):
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
        self.beta = max(img.size[0], img.size[1]) if beta is None else beta
        self.sigma = sigma

        self.kernel = self.get_kernel()


    def patch_indices(self, flatten=False):
        n = 2*self.pr + 1
        indices = np.indices((n, n)) - self.pr
        indices = indices.transpose(1, 2, 0)
        if flatten:
            indices = indices.reshape(-1, 2)
        return indices

    def get_kernel(self):
        pr = self.pr
        x, y = np.mgrid[-pr:pr+1:1, -pr:pr+1:1]
        pos = np.dstack((x, y))
        g = multivariate_normal(mean=np.zeros(2), cov=self.sigma*np.eye(2)).pdf(pos)
        g /= np.sum(g)
        return g

    def fz(self, phi, y, x, bbox_A_t, bbox_A, indices_B, B):
        z = np.array([y, x])
        h = self.patch_indices(flatten=True)  # (p, 2)

        d = z[None, :] - h  # (p, 2)
        p = phi[d[:, 0]-bbox_A_t.y1-1, d[:, 1]-bbox_A_t.x1-1]  # (p, 2)
        p = p + h  # (p, 2)

        # g = gaussian_filter(u_t, sigma=0.5)
        g = self.kernel.flatten()
        u_t = self.array_B[p[:, 0], p[:, 1], :]  # (p, 3)
        u_t = np.inner(g, u_t.T)
        # u_t = u_t.mean(axis=0)

        return u_t

        # exit()

        # M = (p[None, :, :] == indices_B[:, None, :]).astype(int)
        # # print(np.unique(M, return_counts=True))
        # M = M.any(axis=2)
        # M = M.sum(axis=1)
        # # print(M.shape)
        # # print(B.shape)
        # # print(np.unique(M, return_counts=True))
        # SM = M.sum()
        # # print(SM)
        # a = np.inner(M, B.T)/SM
        # # print(a.shape)
        # print(a)
        # # exit()

        # return a

        # print(p)

    def image_update(self, phi, bbox_A_t, bbox_A, indices_B, B):

        u = bbox_A.zeros(3)

        for i, (y, x) in enumerate(bbox_A):
            # print(i)
            u[y-bbox_A.y1-1, x-bbox_A.x1-1, :] = self.fz(phi, y, x, bbox_A_t, bbox_A, indices_B, B)

        return u

    def map_update(self, u, bbox_A, bbox_A_t, n_iter, indices_B, B):
        return self.patch_match(u, bbox_A, bbox_A_t, n_iter, indices_B, B)

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


        xt1, yt1, xt2, yt2 = bbox_A_t.coords
        img_draw.line([(xt1, yt1), (xt2, yt1)], fill=(0, 255, 0))
        img_draw.line([(xt1, yt1), (xt1, yt2)], fill=(0, 255, 0))
        img_draw.line([(xt2, yt1), (xt2, yt2)], fill=(0, 255, 0))
        img_draw.line([(xt1, yt2), (xt2, yt2)], fill=(0, 255, 0))


        whole_u = np.array(B_masked)

        W, H = self.bbox_B.size
        is_patch_in_A_t = np.zeros((H, W)).astype(bool)
        x1, y1, x2, y2 = bbox_A_t.coords
        is_patch_in_A_t[y1:y2+1, x1:x2+1] = True
        is_patch_in_A_t = is_patch_in_A_t.flatten()

        indices_B = np.indices((H, W)).reshape(2, H*W).T
        indices_B = np.delete(indices_B, is_patch_in_A_t, axis=0)

        B = np.array(self.B)

        # phi = self.init_phi(H, W, bbox_A_t)

        for k in range(n_iter):
            print(f'inpainting iter {k}')
            phi = self.map_update(whole_u, bbox_A, bbox_A_t, n_iter_pm, indices_B, B)

            phi_r = phi.reshape(-1, 2)
            assert Bbox(self.pr, self.pr, W-self.pr, H-self.pr).is_inside(phi_r).all()

            u = self.image_update(phi, bbox_A_t, bbox_A, indices_B, B)
            # print(u.shape)
            # exit()
            whole_u[bbox_A.y1:bbox_A.y2+1, bbox_A.x1:bbox_A.x2+1, :] = u


            current_img = Image.fromarray(np.uint8(whole_u))
            # current_img.show()

            draw = ImageDraw.Draw(current_img)
            for i in range(phi.shape[0]):
                for j in range(phi.shape[1]):
                    i2, j2 = phi[i, j]
                    self.draw_center_patch(draw, j2, i2, (255, 0, 0))

            current_img.show()
            exit()


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

    def draw_center_patch(self, draw, x, y, color):
        r = self.pr
        draw.line([x-r, y, x+r, y], fill=color)
        draw.line([x, y-r, x, y+r], fill=color)

    def patch_match(self, u, bbox_A, bbox_A_t, n_iter, indices_B, B):
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

        pr = self.pr

        def D(p1, p2, flip):
            d = np.sum(np.power(p1 - p2, 2), axis=2)
            if flip:
                d1 = d[pr+1:, :].flatten()
                k1 = self.kernel[pr+1:, :].flatten()
                d2 = d[pr, pr+1:].flatten()
                k2 = self.kernel[pr, pr+1:].flatten()

            else:
                d1 = d[:pr, :].flatten()
                k1 = self.kernel[:pr, :].flatten()
                d2 = d[pr, :pr].flatten()
                k2 = self.kernel[pr, :pr].flatten()

            return np.inner(k1, d1) + np.inner(k2, d2)

        # def D(p1, p2):
        #     d = np.sum(np.power(p1 - p2, 2), axis=2).flatten()
        #     k = self.kernel.flatten()

        #     return np.inner(k, d)

        x0, y0, *_ = bbox_A_t.coords
        print(phi.shape)
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
                    y2, x2 = bbox_A_t.outside(y, x+delta)  #phi[y-y0, x-x0+delta, :]  # left/right
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
                    y3, x3 = bbox_A_t.outside(y+delta, x) #phi[y-y0+delta, x-x0, :]  # up/down
                    # y3, x3 = None, None #phi[y-y0, x-x0, :]
                    # print("Edge up", y3, x3, y, x)
                # self.draw_center_patch(draw, x3, y3, (0, 255, 0))

                patch1 = u[y1-pr:y1+pr+1, x1-pr:x1+pr+1, :]
                D1 = D(patch0, patch1, flip)

                if y2 is not None:
                    patch2 = u[y2-pr:y2+pr+1, x2-pr:x2+pr+1, :]
                    # print(y2-pr, y2+pr+1, x2-pr, x2+pr+1)
                    D2 = D(patch0, patch2, flip)
                if y3 is not None:
                    patch3 = u[y3-pr:y3+pr+1, x3-pr:x3+pr+1, :]
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

                i2, j2 = argmin
                # draw.line([(j2-r, i2), (j2+r, i2)], fill=(255, 255, 0))
                # draw.line([(j2, i2-r), (j2, i2+r)], fill=(255, 255, 0))
                # self.draw_center_patch(draw, j2, i2, (255, 255, 0))

                # break
                # if nb > 10:
                #     break

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
                    y_rand, x_rand = np.clip((v0 + radius*eps).astype(int), [pr, pr], [H-pr-1, W-pr-1])

                    if bbox_A_t.is_inside([y_rand, x_rand]).any():
                        k += 1
                        continue

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
