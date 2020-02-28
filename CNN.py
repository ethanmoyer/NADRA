import csv
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import LeakyReLU, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# Retrieves both the post processed training data and the post processed test data.
dataset_test = pd.read_csv('processed_data/post_test_data.csv')
dataset_train = pd.read_csv('processed_data/0.4_50000_train_data.csv')

# Segregrates outputs from both data sets.
y_train = dataset_train['output']
y_test = dataset_test['output']

# Removes outputs from both data sets.
dataset_train = dataset_train.drop(['output', 'sequence'], axis=1)
dataset_test = dataset_test.drop(['output', 'sequence'], axis=1)

# Reshapes train feature set and 
dataset_train_numpy = dataset_train.values
dataset_train = X_train_numpy.reshape(dataset_train.shape[0], 1, dataset_train.shape[1])
y_train_numpy = y_train.values
y_train = y_train_numpy.reshape(y_train.shape[0], 1, 1)

dataset_test_numpy = dataset_test.values
dataset_test = X_test_numpy.reshape(dataset_test.shape[0], 1, dataset_test.shape[1])
y_test_numpy = y_test.values
y_test = y_test_numpy.reshape(y_test.shape[0], 1, 1)

batch_size = X_train.shape[0]
epochs = 20
num_classes = 2

model = Sequential()
model.add(Conv1D(128, kernel_size=3,activation='linear',
                 input_shape=(1, X_train.shape[2]),padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(MaxPooling1D(2,padding='same'))
model.add(Conv1D(256, kernel_size=3, activation='linear',
                 padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2,padding='same'))
model.add(Conv1D(512, kernel_size=3, activation='linear',
                 padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2,padding='same'))
model.add(Conv1D(1024, kernel_size=3, activation='linear',
                 padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2,padding='same'))
# model.add(Flatten())
model.add(Dense(1024, activation='linear'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
model.summary()

model_train = model.fit(X_train, y_train, batch_size=batch_size,
                        epochs=epochs,verbose=1,
                        validation_data=(X_test, y_test))

y_pred = model.predict_classes(X_test)

print([['true negative', 'false positive'],
	['false negative','true positive']])
# precision = (true posiitive)/(predicted positive)
# recall = (true positive)/(total actual positive)
# f1 = 2 * (precision * recall) / (precision + recall)

# Prints confusion matrix.
print(confusion_matrix(y_test, y_pred))

# Prints classification report.
print(classification_report(y_test, y_pred))