import numpy as np
import os

pred_file = r'D:\20230318_SplitGFP_pR_noliposomes_titration\20260318_SplitGFP_proteorhodopsin_noliposomes_droplets__P0_pred.npy'
dat = np.load(pred_file, allow_pickle=True).item()
print('Keys:', list(dat.keys()))

masks = dat.get('masks')
if masks is not None:
    print('masks shape:', masks.shape, 'dtype:', masks.dtype, 'max:', masks.max(), 'nonzero pixels:', np.count_nonzero(masks))
else:
    print('masks: None')

cs = dat.get('channel_segmentations')
if cs:
    print('channel_segmentations keys:', list(cs.keys()))
    for k, v in cs.items():
        if isinstance(v, dict):
            m = v.get('masks')
            if m is not None:
                m = np.asarray(m)
                print(f'  ch{k} masks shape:', m.shape, 'dtype:', m.dtype, 'max:', m.max())
        else:
            print(f'  ch{k}: not a dict, type={type(v)}')
else:
    print('channel_segmentations: None')

print('active_channel_index:', dat.get('active_channel_index'))
