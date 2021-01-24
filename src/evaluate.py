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

    def _retrieve_paths(self):
        walk = os.walk(self.root)

        print(walk)

    def evaluate(self):
        pass


if __name__ == '__main__':
    Examiner(root='to_evaluate').evaluate()

