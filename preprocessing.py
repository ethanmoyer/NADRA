import csv
import pandas as pd

# Retrieves both the training data and the test data.
dataset_train = pd.read_csv('data/0.4_50000_train_data.csv')
dataset_test = pd.read_csv('data/test_data.csv')

# Normalizes base features in both data sets.
dataset_train.loc[:, "nuc_1":"nuc_46"] = dataset_train.loc[: , "nuc_1":"nuc_46"] / 4
dataset_test.loc[:, "nuc_1":"nuc_46"] = dataset_test.loc[: , "nuc_1":"nuc_46"] / 4

# Normalizes length feature in both data sets.
dataset_train.loc[:, "length"] = dataset_train.loc[:, "length"] / dataset_train.loc[:, 'length'].max()
dataset_test.loc[:, "length"] = dataset_test.loc[:, "length"] / dataset_test.loc[:, 'length'].max()

# Normalizes en feature in both data setsl
dataset_train.loc[:, "en"] = dataset_train.loc[:, "en"] / dataset_train.loc[:, 'en'].max()
dataset_test.loc[:, "en"] = dataset_test.loc[:, "en"] / dataset_test.loc[:, 'en'].max()

# Writes 
dataset_train.to_csv('processed_data/post_0.4_50000_train_data.csv')
dataset_test.to_csv('processed_data/post_test_data.csv')