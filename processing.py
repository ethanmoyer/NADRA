import csv
import pandas as pd

# Retrieves both the training data and the test data.
taining_data_loc = str(input('Enter the location of the training data: '))
if (taining_data_loc.find("data") < 0 or taining_data_loc.find("train") < 0):
	print("Please enter the location of a training data set from the data directory.")
	exit()
dataset_train = pd.read_csv(taining_data_loc)

test_data_loc = str(input('Enter the location of the test data: '))
if (test_data_loc.find("data") < 0 or test_data_loc.find("test") < 0):
	print("Please enter the location of a test data set from the data directory.")
	exit()
dataset_test = pd.read_csv(test_data_loc)

# Normalizes base features in both data sets.
dataset_train.loc[:, "nuc_1":"nuc_46"] = dataset_train.loc[: , "nuc_1":"nuc_46"] / 4
dataset_test.loc[:, "nuc_1":"nuc_46"] = dataset_test.loc[: , "nuc_1":"nuc_46"] / 4

# Normalizes length feature in both data sets.
dataset_train.loc[:, "length"] = dataset_train.loc[:, "length"] / dataset_train.loc[:, 'length'].max()
dataset_test.loc[:, "length"] = dataset_test.loc[:, "length"] / dataset_test.loc[:, 'length'].max()

# Normalizes en feature in both data sets.
dataset_train.loc[:, "en"] = dataset_train.loc[:, "en"] / dataset_train.loc[:, 'en'].max()
dataset_test.loc[:, "en"] = dataset_test.loc[:, "en"] / dataset_test.loc[:, 'en'].max()

# Writes the normalized training and test data to the processed_data directory.
dataset_train.to_csv('processed_data/' + taining_data_loc[(taining_data_loc.find('/') + 1):], index=False)
dataset_test.to_csv('processed_data/' + test_data_loc[(test_data_loc.find('/') + 1):], index=False)