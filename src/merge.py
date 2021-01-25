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

df['SNR'] = df['SNR_mask_only']
df['PSNR'] = df['PSNR_mask_only']
df = df.drop(['SNR_mask_only', 'PSNR_mask_only'], axis=1)

df['img'] = df['path'].str.split('/', expand=True)[1]
print(df)

df = df.set_index(['method', 'img'], verify_integrity=True)
print(df)

df_avg = df.groupby('method').mean().round(1)

print(df_avg)

df_avg.to_csv(os.path.join(root, 'average.csv'))
df_avg.to_latex(os.path.join(root, 'average.tex'))
df_avg.to_pickle(os.path.join(root, 'average.pickle'))

# Ranks
df_ranks = df.copy()
df_ranks['D_coherence'] = -df_ranks['D_coherence']
dfgb = df_ranks.groupby('img')
print(dfgb)
df_ranks = dfgb.rank(method='dense', ascending=False)

print(df_ranks)
df_ranks.to_csv(os.path.join(root, 'ranks_all.csv'))
df_ranks.to_latex(os.path.join(root, 'ranks_all.tex'))
df_ranks.to_pickle(os.path.join(root, 'ranks_all.pickle'))

df_ranks_avg = df_ranks.groupby('method').mean()

df_ranks_avg['Mean'] = df_ranks_avg.mean(axis=1).round(1)

print(df_ranks_avg)
df_ranks_avg.to_csv(os.path.join(root, 'ranks_avg.csv'))
df_ranks_avg.to_latex(os.path.join(root, 'ranks_avg.tex'))
df_ranks_avg.to_pickle(os.path.join(root, 'ranks_avg.pickle'))


