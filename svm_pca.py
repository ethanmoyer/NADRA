import csv
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
import pandas as pd

# Retrieves both the post processed training data and the post processed test data.
taining_data_loc = str(input('Enter the location of the processed training data: '))
if (taining_data_loc.find("processed_data") < 0 or taining_data_loc.find("train") < 0):
	print("Please enter the location of a training data set from the data directory.")
	exit()
dataset_train = pd.read_csv(taining_data_loc)

test_data_loc = str(input('Enter the location of the processed test data: '))
if (test_data_loc.find("processed_data") < 0 or test_data_loc.find("test") < 0):
	print("Please enter the location of a test data set from the data directory.")
	exit()
dataset_test = pd.read_csv(test_data_loc)

# Segregrates outputs from both data sets.
y_train = dataset_train['output']
y_test = dataset_test['output']

# Removes outputs from both data sets.
dataset_train = dataset_train.drop(['output', 'sequence'], axis=1)
dataset_test = dataset_test.drop(['output', 'sequence'], axis=1)

# Generates PCA based on training data and transforms test data.
# https://stackabuse.com/implementing-pca-in-python-with-scikit-learn/
pca = PCA(n_components=2)
dataset_train = pca.fit_transform(dataset_train)
dataset_test = pca.transform(dataset_test)

# Creates the SVM model
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
svclassifier = SVC(kernel='poly')

# Fits the SVM model according to the given train data
svclassifier.fit(dataset_train, y_train)

# Generates output predicitons based on test data.
y_pred = svclassifier.predict(dataset_test)

print([['true negative', 'false positive'],
	['false negative','true positive']])
# precision = (true posiitive)/(predicted positive)
# recall = (true positive)/(total actual positive)
# f1 = 2 * (precision * recall) / (precision + recall)

# Prints confusion matrix.
print(confusion_matrix(y_test, y_pred))

# Prints classification report.
print(classification_report(y_test, y_pred))

# Plot Decision Region using mlxtend's awesome plotting function
# https://stackoverflow.com/questions/43284811/plot-svm-with-matplotlib
plot_decision_regions(X=dataset_train, 
	y=y_train.values.astype(np.integer), clf=svclassifier, legend=2)

# Update plot object with X/Y axis labels and Figure Title
plt.xlabel("component 1", size=12)
plt.ylabel("component 2", size=12)
plt.title('SVM Decision Region Boundary', size=16)

dataset_test['output'] = y_test;
dataset_test['predicted'] = y_pred;
dataset_test.to_csv("analysis_data/SVM_PCA_" + test_data_loc[(test_data_loc.find('/') + 1):], index=False)

print("PCA SVMs analysis finished.")
