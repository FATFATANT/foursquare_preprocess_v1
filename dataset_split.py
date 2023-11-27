import numpy as np
import pandas as pd
import os

dataset = 1
train_df = []
val_df = []
test_df = []
if dataset == 0:
    city = 'NYC'
elif dataset == 1:
    city = 'SIN'
else:
    city = 'TKY'


def dataset_split(df):
    all_traj = df['trajectory_id'].unique()
    train_id = all_traj[:int(len(all_traj) * 0.6)]
    val_id = all_traj[int(len(all_traj) * 0.6):int(len(all_traj) * 0.8)]
    test_id = all_traj[int(len(all_traj) * 0.8):]
    train_df.append(df[df['trajectory_id'].isin(train_id)].copy())
    val_df.append(df[df['trajectory_id'].isin(val_id)].copy())
    test_df.append(df[df['trajectory_id'].isin(test_id)].copy())


in_path = os.path.join('out', city, 'checkins_v1.csv')
out_train = os.path.join('out', city, 'train.csv')
out_val = os.path.join('out', city, 'val.csv')
out_test = os.path.join('out', city, 'test.csv')
raw = pd.read_csv(in_path)

raw.groupby('user_id').apply(lambda x: dataset_split(x))

train = pd.concat(train_df).reset_index(drop=True)
val = pd.concat(val_df).reset_index(drop=True)
test = pd.concat(test_df).reset_index(drop=True)

"""
目前来说，总的POI=独立POI+集合POI中的独立POI+构造出的集合POI
而独立POI+集合POI中的独立POI=actual_poi_id
且独立POI+构造出的集合POI=check_in_poi_id
因此我将其全部取出再取并集即可
"""
all_poi_in_train = np.union1d(train['actual_poi_id'].unique(), train['check_in_poi_id'].unique())

train.to_csv(out_train, index=False)
val.to_csv(out_val, index=False)
test.to_csv(out_test, index=False)
