"""Use this file to evaluate inpainting results."""
import os
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
import argparse

from time import time


class Examiner():

    def __init__(self, root, ext='jpg'):
        """Init."""
        if not os.path.exists(root):
            raise ValueError(f'Path doesn\'t exist: {root}')

        self.root = root
        self.ext = ext

        # Required filenames
        self.img_filename = f'img.{ext}'
        self.mask_filename = f'mask.{ext}'
        self.inpainted_filename = f'inpainted.{ext}'

        self.required_filenames = set([
            self.img_filename,
            self.mask_filename,
            self.inpainted_filename
        ])

        self.img_folder_paths = self._retrieve_paths()

        print(f'{len(self.img_folder_paths)} images found.')

    def _retrieve_paths(self):
        walk = os.walk(self.root)

        img_folder_paths = []

        for folder, subfolders, filenames in walk:
            # Keep only leaf folders
            if subfolders:
                continue

            # Check if required files are present
            if not self.required_filenames.issubset(filenames):
                print(f'Warning: dir "{folder}"" doesnt contain all of '
                      f'the required filenames {self.required_filenames}.')
                # continue

            img_folder_paths.append(folder)

        return img_folder_paths

    def evaluate(self, pr, stride_out, stride_in, compute_completeness=False):

        rows = []

        for path in self.img_folder_paths:
            print(f'Evaluating "{path}"')
            img = Image.open(os.path.join(path, self.img_filename))
            mask = Image.open(os.path.join(path, self.mask_filename))
            inpainted = Image.open(os.path.join(path, self.inpainted_filename))

            SNR = self.SNR(img, inpainted, mask=None)
            SNR_mask = self.SNR(img, inpainted, mask=mask)
            PSNR = self.PSNR(img, inpainted, mask=None)
            PSNR_mask = self.PSNR(img, inpainted, mask=mask)

            SSIM = self.SSIM(img, inpainted, mask, pr, stride_in)

            D_coherence, D_complete = self.D_BDS(img, inpainted, mask, pr, stride_out, stride_in, compute_completeness)

            row = [path, SNR, SNR_mask, PSNR, PSNR_mask, SSIM, D_coherence]
            if compute_completeness:
                D_DBS = D_coherence + D_complete
                row.append(D_complete)
                row.append(D_DBS)

            rows.append(row)

        cols = [
            'path',
            'SNR',
            'SNR_mask_only',
            'PSNR',
            'PSNR_mask_only',
            'SSIM',
            'D_coherence',
        ]

        if compute_completeness:
            cols.apppend('D_complete')
            cols.append('D_DBS')

        return pd.DataFrame(rows, columns=cols)

    @staticmethod
    def SNR(img, img_inpainted, mask=None):
        """Compute singal to noise ratio."""
        img = np.array(img)
        img_inpainted = np.array(img_inpainted)

        if mask:
            mask = ~np.array(mask).astype(bool)
            idx = np.where(mask)

            # Crop images where the mask is
            img = img[idx]
            img_inpainted = img_inpainted[idx]

        D1 = np.power(img_inpainted, 2)
        D2 = np.power(img - img_inpainted, 2)

        SNR = np.sum(D1)/np.sum(D2)

        return SNR

    @staticmethod
    def PSNR(img, img_inpainted, mask=None):
        """Compute the peak signal-to-noise ratio."""
        img = np.array(img)
        img_inpainted = np.array(img_inpainted)

        if mask:
            mask = ~np.array(mask).astype(bool)
            idx = np.where(mask)

            # Crop images where the mask is
            img = img[idx]
            img_inpainted = img_inpainted[idx]

        D = np.power(img - img_inpainted, 2)
        MSE = np.mean(D.flatten())
        MAX = np.max(img)
        PSNR = 20*np.log10(MAX) - 10*np.log10(MSE)

        return PSNR

    @staticmethod
    def D_BDS(img, img_inpainted, mask, pr, stride_out, stride_in, compute_completeness):
        """Compute coherence & complete dist defined in PatchMatch paper."""
        img = np.array(img).astype(np.uint8)
        img_inpainted = np.array(img_inpainted).astype(np.uint8)
        mask = ~np.array(mask).astype(bool)
        idx = np.where(mask)

        n = 2*pr+1
        N = n**2

        H, W, C = img.shape

        h = max(idx[0]) - min(idx[0]) + 1
        w = max(idx[1]) - min(idx[1]) + 1
        x0 = idx[1][0]
        y0 = idx[0][0]

        print(f'\tStride out: {stride_out}, stride in: {stride_in}, patch radius: {pr}')
        print(f'\tRetrieving ({n}x{n}) patches outside inpainting area...', end=' ')
        patches_in_S = []
        for y in range(pr, H-pr, stride_out):
            for x in range(pr, W-pr, stride_out):
                if mask[y, x].any():
                    continue  # ignore patches in the mask

                patches_in_S.append(img[y-pr:y+pr+1, x-pr:x+pr+1, :].reshape(N, -1))

        print(f'Retrieved {len(patches_in_S)}.')
        patches_in_S = np.stack(patches_in_S)
        patches_in_S = patches_in_S.reshape(-1, N*C)

        print(f'\tRetrieving ({n}x{n}) patches inside inpainting area...', end=' ')
        patches_in_T = []
        for y in range(y0, y0+h+1, stride_in):
            for x in range(x0, x0+w+1, stride_in):
                p = img_inpainted[y-pr:y+pr+1, x-pr:x+pr+1, :].reshape(N, -1)
                patches_in_T.append(p)

        print(f'Retrieved {len(patches_in_T)}.')
        patches_in_T = np.stack(patches_in_T)
        patches_in_T = patches_in_T.reshape(-1, N*C)

        # Coherence distance
        print('\tBuilding kdtree...')
        kdtree = KDTree(patches_in_S)

        print('\tFinding nearest neighbors of patches inside inpainting area...')
        t0 = time()
        D_coherence = kdtree.query(patches_in_T)[0]
        D_coherence = np.mean(D_coherence)
        print(f'\tDone {time() - t0:.2f}s')

        if compute_completeness:
            # Completeness distance
            kdtree = KDTree(patches_in_T)

            print('\tFinding nearest neighbors of patches outside inpainting area...')
            t0 = time()
            D_complete = kdtree.query(patches_in_S)[0]
            print(f'\tDone {time() - t0:.2f}s')

            D_complete = np.mean(D_complete)

        else:
            D_complete = None

        return D_coherence, D_complete

    @staticmethod
    def _ssim(patch1, patch2, L, k1=0.01, k2=0.03):
        mu1 = np.mean(patch1)
        mu2 = np.mean(patch2)
        var1 = np.var(patch1)
        var2 = np.var(patch2)
        cov = np.mean((patch1 - mu1)*(patch2 - mu2))

        c1 = (k1*L)**2
        c2 = (k2*L)**2

        a = (2*mu1*mu2 + c1)*(2*cov + c2)
        b = (mu1**2 + mu2**2 + c1)*(var1 + var2 + c2)

        SSIM = a/b

        return SSIM

    @staticmethod
    def SSIM(img, img_inpainted, mask, pr, stride_in, L=255):
        img = img.convert('L')
        img = np.array(img)
        img_inpainted = img_inpainted.convert('L')
        img_inpainted = np.array(img_inpainted)

        mask = ~np.array(mask).astype(bool)
        idx = np.where(mask)

        n = 2*pr+1
        N = n**2

        H, W = img.shape

        h = max(idx[0]) - min(idx[0]) + 1
        w = max(idx[1]) - min(idx[1]) + 1
        x0 = idx[1][0]
        y0 = idx[0][0]

        SSIMs = []
        for y in range(y0, y0+h+1, stride_in):
            for x in range(x0, x0+w+1, stride_in):
                p_inpainted = img_inpainted[y-pr:y+pr+1, x-pr:x+pr+1].flatten()
                p_real = img[y-pr:y+pr+1, x-pr:x+pr+1].flatten()
                SSIMs.append(Examiner._ssim(p_inpainted, p_real, L))

        return np.mean(SSIMs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str,
                        help='Folder containing the subfolders to evaluate.')
    parser.add_argument('--ext', type=str, default='jpg',
                        help='Image extension.')
    parser.add_argument('--stride-in', type=int, default=1,
                        help='Stride inside the inpainting area.')
    parser.add_argument('--stride-out', type=int, default=1,
                        help='Stride outside the inpainting area.')
    parser.add_argument('--pr', type=int, default=2,
                        help='Patch radius (A 0-radius patch is a pixel).')
    parser.add_argument('--complete', nargs='?', type=bool, const=True,
                        help='Whether to compute the completeness distance.')

    args = parser.parse_args()

    root = os.path.join(args.root)
    df = Examiner(root=root, ext=args.ext).evaluate(args.pr, args.stride_in, args.stride_out, args.complete)

    df.to_csv(os.path.join(root, 'results.csv'))
    df.to_latex(os.path.join(root, 'results.tex'))
    df.to_pickle(os.path.join(root, 'results.pickle'))

    print(df)

    print(f'Results were dumped in "{args.root}".')
