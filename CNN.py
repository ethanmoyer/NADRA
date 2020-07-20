# source ./venv/bin/activate
# python3.7
# 
# processed_data/test_data.csv
# processed_data/0.5_25000_train_data.csv
import csv
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, BatchNormalization, ReLU

# ---

# Retrieves both the post processed training data, the post processed test data, and the post processed validation data.
taining_data_location = 'processed_data/train_data.csv'
#print(f'Default reference file location: {taining_data_location}')
train_ans = str(input('Would you like to use the default training data? [yes/no]: '))
if train_ans.lower() == 'no':
	taining_data_location = str(input('Enter the location of the processed training data: '))
	if (taining_data_location.find("processed_data") < 0 or taining_data_location.find("train") < 0):
		print("Please enter the location of a training data set from the processed_sdata directory.")
		exit()
dataset_train = pd.read_csv(taining_data_location)

# ---
test_data_location = 'processed_data/test_data.csv'
#print(f'Default reference file location: {test_data_location}')
test_ans = str(input('Would you like to use the default test data? [yes/no]: '))
if test_ans.lower() == 'no':
	test_data_location = str(input('Enter the location of the processed test data: '))
	if (test_data_location.find("processed_data") < 0 or test_data_location.find("test") < 0):
		print("Please enter the location of a test data set from the processed_data directory.")
		exit()
dataset_test = pd.read_csv(test_data_location)

# ---
val_data_location = 'processed_data/val_data.csv'
#print(f'Default reference file location: {val_data_location}')
test_ans = str(input('Would you like to use the default validation data? [yes/no]: '))
if test_ans.lower() == 'no':
	val_data_location = str(input('Enter the location of the processed validation data: '))
	if (val_data_location.find("processed_data") < 0 or val_data_location.find("val") < 0):
		print("Please enter the location of a validation data set from the processed_data directory.")
		exit()
dataset_validation = pd.read_csv(val_data_location)

# Segregrates outputs from both data sets.
y_train = dataset_train['output']
y_test = dataset_test['output']
y_val = dataset_validation['output']

# Removes outputs from both data sets.
dataset_train = dataset_train.drop(['output', 'sequence'], axis=1)
dataset_test = dataset_test.drop(['output', 'sequence'], axis=1)
dataset_validation = dataset_validation.drop(['output', 'sequence'], axis=1)

# Saves test set.
X_test = dataset_test

# Saves train set.
X_train = dataset_train

# Reformats train set, test set, validation set and their respective outputs.
X_train_numpy = X_train.values
X_train = X_train_numpy.reshape(X_train.shape[0], 1, X_train.shape[1])
y_train_numpy = y_train.values
y_train = y_train_numpy.reshape(y_train.shape[0], 1, 1)

X_test_numpy = X_test.values
X_test = X_test_numpy.reshape(X_test.shape[0], 1, X_test.shape[1])
y_test_numpy = y_test.values
y_test = y_test_numpy.reshape(y_test.shape[0], 1, 1)

val_numpy = dataset_validation.values
dataset_validation = val_numpy.reshape(dataset_validation.shape[0], 1, dataset_validation.shape[1])
dataset_y_numpy = y_val.values
y_val = dataset_y_numpy.reshape(y_val.shape[0], 1, 1)

# Initializes the main variables for the model.
batch_size = dataset_train.shape[0]
epochs = 50
num_classes = 2

# Builds the model
# Make the model deeper and narrower
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3,activation='softplus', input_shape=(1, X_train.shape[2]),kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01), padding='same',strides=3))
model.add(MaxPooling1D(pool_size=2, padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Conv1D(filters=128, kernel_size=3, activation='linear',
			padding='same', strides=3))
model.add(MaxPooling1D(pool_size=2, padding='same'))
model.add(BatchNormalization())

model.add(Conv1D(filters=256, kernel_size=3, activation='linear',
			padding='same', strides=3))
model.add(MaxPooling1D(pool_size=2, padding='same'))
model.add(BatchNormalization())

model.add(Conv1D(filters=512, kernel_size=3, activation='linear',
			padding='same', strides=3))
model.add(MaxPooling1D(pool_size=2, padding='same'))
model.add(BatchNormalization())

model.add(Conv1D(filters=1024, kernel_size=3, activation='linear',
			padding='same', strides=3))
model.add(MaxPooling1D(pool_size=2, padding='same'))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(1024, activation='relu'))            
model.add(Dense(num_classes, activation='softmax')

# Compiles the model
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy', 'binary_crossentropy'])

# Summarizes the layers in the model.
model.summary()

# Cross validates with validation data.
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,verbose=1, validation_data=(dataset_validation, y_val))

# Predicts using the model on a new data set.
#y_pred = model.predict_classes(X_test)

print([['true negative', 'false positive'],
	['false negative','true positive']])
# precision = (true posiitive)/(predicted positive)
# recall = (true positive)/(total actual positive)
# f1 = 2 * (precision * recall) / (precision + recall)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model absolute loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('figures/test.png')
plt.clf()


# Prints confusion matrix.
#print(confusion_matrix(y_test_numpy, y_pred))

# Prints classification report.
#print(classification_report(y_test_numpy, y_pred))

# Rewrites output and preducted values to the test data set and saves it for a file for further analysis.
#dataset_test['output'] = y_test_numpy;
#dataset_test['predicted'] = y_pred;
#dataset_test.to_csv("analysis_data/CNNs_" + test_data_loc[(test_data_loc.find('/') + 1):], index=False)

print("CNNs analysis finished.")

