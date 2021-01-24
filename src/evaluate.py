"""Use this file to evaluate inpainting results."""
import os
from PIL import Image


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

            img_folder_paths.append(folder)

        return img_folder_paths

    def evaluate(self):
        pass


if __name__ == '__main__':
    Examiner(root='to_evaluate').evaluate()

