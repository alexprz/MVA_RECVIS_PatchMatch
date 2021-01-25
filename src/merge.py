import os
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
import argparse

from time import time

parser = argparse.ArgumentParser()
parser.add_argument('root', type=str,
                    help='Folder containing the subfolders to evaluate.')

args = parser.parse_args()

root = args.root
ext = '.pickle'
df_names = ['pepsi', 'patchmatch', 'semantic', 'sparse']
dfs = []

for name in df_names:
    sub_df = pd.read_pickle(os.path.join(root, name+ext))
    sub_df['method'] = name
    dfs.append(sub_df)

df = pd.concat(dfs, axis=0)

print(df)
