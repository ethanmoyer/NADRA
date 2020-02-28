import csv
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Retrieves both the post processed training data and the post processed test data.
dataset_test = pd.read_csv('processed_data/post_test_data.csv')
dataset_train = pd.read_csv('processed_data/0.4_50000_train_data.csv')

# Segregrates outputs from both data sets.
y_train = dataset_train['output']
y_test = dataset_test['output']

# Removes outputs from both data sets.
dataset_train = dataset_train.drop(['output', 'sequence'], axis=1)
dataset_test = dataset_test.drop(['output', 'sequence'], axis=1)


# Creates the SVM model
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
svclassifier = SVC(kernel='poly')

# Fits the SVM model according to the given train data
svclassifier.fit(dataset_train, y_train)

# Generates output predicitons based on test data.
y_pred = svclassifier.predict(X_test)

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