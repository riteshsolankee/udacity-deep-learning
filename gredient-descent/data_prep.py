import pandas as pd
from functools import reduce
import numpy as np


def prepare_data(df):
    df_dummy = pd.get_dummies(df['rank'], prefix='rank', columns=['rank'])
    normalize_df = (df.drop(['rank', 'admit'], axis=1) - df.drop(['rank', 'admit'], axis=1).mean())/df.drop(['rank', 'admit'], axis=1).std()
    dfs = [df['admit'].to_frame(), normalize_df, df_dummy]

    final_df = reduce(lambda l,r: pd.merge(l,r, left_index=True, right_index=True), dfs)

    # print(final_df.head(10))

    # x = np.random.rand(100, 5)
    # print(final_df.shape)
    indices = np.random.permutation(final_df.shape[0])
    # print(indices)
    training_idx, test_idx = indices[:int(len(final_df)*0.9)], indices[int(len(final_df)*0.9):]
    # print("\ntraining_idx:", len(training_idx))
    # print("\ntest_idx:", len(test_idx))

    training, test = final_df.iloc[training_idx,:], final_df.iloc[test_idx,:]
    # print(training.head(5))
    # print(test.head(5))

    features, target, features_test, target_test = \
        training.drop(['admit'], axis=1), training['admit'], test.drop(['admit'], axis=1), test['admit']
    return features, target, features_test, target_test
