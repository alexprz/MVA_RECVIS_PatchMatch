# PatchMatch

## Inpainting evaluation

This is done in the `evaluation.py` file by passing the following arguments.
```
usage: evaluate.py [-h] [--ext EXT] [--stride-in STRIDE_IN]
                   [--stride-out STRIDE_OUT] [--pr PR] [--complete [COMPLETE]]
                   root

positional arguments:
  root                  Folder containing the subfolders to evaluate.

optional arguments:
  -h, --help            show this help message and exit
  --ext EXT             Image extension.
  --stride-in STRIDE_IN
                        Stride inside the inpainting area.
  --stride-out STRIDE_OUT
                        Stride outside the inpainting area.
  --pr PR               Patch radius (A 0-radius patch is a pixel).
  --complete [COMPLETE]
                        Whether to compute the completeness distance.
```

First create a root folder in the `src\` directory. The architecture
inside this folder must be:
```
src/
    root/
        image1/
            img.ext
            inpainted.ext
            mask.ext
        image2/
            img.ext
            inpainted.ext
            mask.ext
        ...
```
The default extension is `jpg` but you can change it passing the `--ext`
parameter.

### Usage example:
```
    python evaluate.py --root to_evaluate --ext png
```
