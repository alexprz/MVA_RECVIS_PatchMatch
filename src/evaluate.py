"""Use this file to evaluate inpainting results."""
import os
from PIL import Image
import pandas as pd
import numpy as np


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
            img = Image.open(os.path.join(path, self.img_filename))
            mask = Image.open(os.path.join(path, self.mask_filename))
            inpainted = Image.open(os.path.join(path, self.inpainted_filename))

            SNR = self.SNR()
            PSNR = self.PSNR(img, inpainted, mask=None)
            PSNR_mask = self.PSNR(img, inpainted, mask=mask)

            row = [path, SNR, PSNR, PSNR_mask]
            rows.append(row)

        return pd.DataFrame(rows, columns=['path', 'SNR', 'PSNR', 'PSNR_mask_only'])

    @staticmethod
    def SNR():
        return 0

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


if __name__ == '__main__':
    df = Examiner(root='to_evaluate').evaluate()

    print(df)

