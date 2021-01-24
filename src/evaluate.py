"""Use this file to evaluate inpainting results."""
import os
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree

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
                print(f'Warning: dir {folder} ignored because doesnt contain '
                      f'the required filenames {self.required_filenames}.')
                continue

            img_folder_paths.append(folder)

        return img_folder_paths

    def evaluate(self):

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

            D_coherence, D_complete = self.D_BDS(img, inpainted, mask, 2)
            D_DBS = D_coherence + D_complete

            row = [path, SNR, SNR_mask, PSNR, PSNR_mask, D_coherence, D_complete, D_DBS]
            rows.append(row)

        return pd.DataFrame(rows, columns=[
            'path',
            'SNR',
            'SNR_mask_only',
            'PSNR',
            'PSNR_mask_only',
            'D_coherence',
            'D_complete',
            'D_DBS',
        ])

    @staticmethod
    def SNR(img, img_inpainted, mask=None):
        """Compute singal to noise ratio."""
        img = np.array(img)
        img_inpainted = np.array(img_inpainted)

        if mask:
            mask = np.array(mask).astype(bool)
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
            mask = np.array(mask).astype(bool)
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
    def D_BDS(img, img_inpainted, mask, pr, stride_out=5, stride_in=2):
        """Compute coherence & complete dist defined in PatchMatch paper."""
        img = np.array(img).astype(np.uint8)
        img_inpainted = np.array(img_inpainted).astype(np.uint8)
        mask = np.array(mask).astype(bool)
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
        print(f'\tDone {time() - t0:.2f}s')

        # Completeness distance
        kdtree = KDTree(patches_in_T)

        print('\tFinding nearest neighbors of patches outside inpainting area...')
        t0 = time()
        D_complete = kdtree.query(patches_in_S)[0]
        print(f'\tDone {time() - t0:.2f}s')

        return np.mean(D_coherence), np.mean(D_complete)


if __name__ == '__main__':
    root = os.path.join('to_evaluate')
    df = Examiner(root=root).evaluate()

    df.to_csv(root+'results.csv')
    df.to_latex(root+'results.tex')
    df.to_pickle(root+'results.pickle')

    print(df)

