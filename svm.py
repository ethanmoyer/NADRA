import csv
from sklearn import svm
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

