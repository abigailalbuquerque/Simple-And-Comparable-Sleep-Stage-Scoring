# Import libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import wave
import pylab
from pathlib import Path
from scipy import signal
from scipy.io import wavfile
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, cohen_kappa_score
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import math

# Unzip imported training and testing dataset
!unzip /content/drive/MyDrive/Colab_Data/training_set_00_17_Sc.zip
!unzip /content/drive/MyDrive/Colab_Data/testing_set_18_19_Sc.zip

# Load X_train and Y_train
subject_id_list = np.array(["SC4001E0", "SC4002E0", "SC4011E0", "SC4012E0", "SC4021E0", "SC4022E0",
                            "SC4031E0", "SC4032E0", "SC4041E0", "SC4042E0", "SC4051E0", "SC4052E0",
                            "SC4061E0", "SC4062E0", "SC4071E0", "SC4072E0", "SC4081E0", "SC4082E0",
                            "SC4091E0", "SC4092E0", "SC4101E0", "SC4102E0", "SC4111E0", "SC4112E0",
                            "SC4121E0", "SC4122E0", "SC4131E0",
                            "SC4141E0", "SC4142E0", "SC4151E0", "SC4152E0", "SC4161E0", "SC4162E0",
                            "SC4171E0", "SC4172E0"])
X_train_raw = np.empty((0, 138, 28))
Y_train_raw = np.empty((0))

for subject_id in subject_id_list:
  print('\r', f"loading...: {subject_id}", end='')
  X_file_name = 'training_set_00_17_Sc/' + 'X_' + subject_id + '.mat'
  Y_file_name = 'training_set_00_17_Sc/' + 'Y_' + subject_id + '.mat'

  # extract data from X
  mat = loadmat(X_file_name)
  mat_3d = mat['Sc_ds']
  numpy_3d = np.array(mat_3d)
  numpy_3d_reshape = numpy_3d.transpose((2,1,0))
  X_train_raw = np.concatenate((X_train_raw, numpy_3d_reshape), axis=0)

  # extract data from Y
  Y_true = loadmat(Y_file_name)
  Y_train = Y_true['y_true_list']
  Y_train = Y_train.flatten()
  Y_train_raw = np.append(Y_train_raw, Y_train)

# print the shape to check size
print('  Done')
print(X_train_raw.shape)
print(Y_train_raw.shape)

# Load X_test and Y_test
subject_id_list = np.array(["SC4181E0", "SC4182E0", "SC4191E0", "SC4192E0"])
X_test_raw = np.empty((0, 138, 28))
Y_test_raw = np.empty((0))

for subject_id in subject_id_list:
  print('\r', f"loading...: {subject_id}", end='')
  X_file_name = 'testing_set_18_19_Sc/' + 'X_' + subject_id + '.mat'
  Y_file_name = 'testing_set_18_19_Sc/' + 'Y_' + subject_id + '.mat'

  # extract data from X
  mat = loadmat(X_file_name)
  mat_3d = mat['Sc_ds']
  numpy_3d = np.array(mat_3d)
  numpy_3d_reshape = numpy_3d.transpose((2,1,0))
  X_test_raw = np.concatenate((X_test_raw, numpy_3d_reshape), axis=0)

  # extract data from Y
  Y_true = loadmat(Y_file_name)
  Y_test = Y_true['y_true_list']
  Y_test = Y_test.flatten()
  Y_test_raw = np.append(Y_test_raw, Y_test)

# print the shape to check size
print('  Done')
print(X_test_raw.shape)
print(Y_test_raw.shape)

# Expand the raw value e^8
X_train_raw = X_train_raw*(10**8)
X_test_raw = X_test_raw*(10**8)

# Oversampling
def class_info(Y_train, Y_test):
  # Get the number of different classes from true and pred
  print('         Train  Test')
  unique, true_counts = np.unique(Y_train, return_counts=True)
  unique, pred_counts = np.unique(Y_test, return_counts=True)
  print(np.asarray((['Wake', 'N1', 'N2', 'N3', 'REM'], true_counts, pred_counts)).T)

# get the class_info before oversampling
class_info(Y_train_raw, Y_test_raw)

# input the raw X and Y, return the oversampled X and Y
def oversampling(X_raw, Y_raw):
  X = X_raw
  y = Y_raw

  y = y.astype(int)
  max_size = np.max(np.bincount(y))
  size_diffs = [max_size - np.sum(y == i) for i in range(5)]

  for i in range(5):
    if size_diffs[i] > 0:
        idxs = np.where(y == i)[0]
        n_dup = size_diffs[i]
        idxs_to_dup = np.random.choice(idxs, size=n_dup, replace=True)
        X = np.concatenate((X, X[idxs_to_dup]), axis=0)
        y = np.concatenate((y, np.full(n_dup, i)), axis=0)

  return X, y

X_oversampled_train, Y_oversampled_train = oversampling(X_train_raw, Y_train_raw)

# get the class_info after oversampling
class_info(Y_oversampled_train, Y_test_raw)

# Prepare the train valid test ds
x_train, x_valid, y_train, y_valid = train_test_split(X_oversampled_train, Y_oversampled_train, test_size=0.2, shuffle=True)
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((X_test_raw, Y_test_raw)).batch(32)

# Create CNN model
IMAGE_HEIGHT = 138
IMAGE_WIDTH = 28
N_CHANNELS = 1
N_CLASSES = 5

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNELS)))
model.add(tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu'))
model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Flatten()) # reshapes the tensor to have a shape that is equal to the number of elements contained in the tensor
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(N_CLASSES, activation='softmax'))

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.RMSprop(),
    metrics=['accuracy'],
)


# Train model for 10 epochs, capture the history
history = model.fit(train_ds, epochs=10, validation_data=valid_ds)
model.summary()

# Evaluate the model with test_set
evaluation_set = test_ds
predictions = model.predict(evaluation_set)
label_pred = np.array([])
for sub in predictions:
  max_idx = np.argmax(sub)
  label_pred = np.append(label_pred, max_idx)
test_label = np.concatenate([y for x, y in evaluation_set], axis=0)

# Get confusion matrix and plot out
cm = confusion_matrix(test_label, label_pred)

# Get acc, precision, recall, and f1
acc = accuracy_score(test_label, label_pred)
print('Overall Accuracy: ', acc.round(3))
precision = precision_score(test_label, label_pred, average=None)
print('Precision per-class:', precision.round(3))
recall = recall_score(test_label, label_pred, average=None)
print('Recall per-class:', recall.round(3))
f1score = f1_score(test_label, label_pred, average=None)
print('F1 per-class:', f1score.round(3))
f1weighted = f1_score(test_label, label_pred, average='weighted')   # conpute the f1 for each label and returns the average considering the proportion of each label
print('F1(weighted):', f1weighted.round(3))

# Plot the confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt
ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['Wake', 'N1', 'N2', 'N3', 'REM']); ax.yaxis.set_ticklabels(['Wake', 'N1', 'N2', 'N3', 'REM']);
# move labels to the top
ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
ax.xaxis.set_label_position('top')

# Get the number of different classes from true and pred
print('         True  Pred')
unique, true_counts = np.unique(test_label, return_counts=True)
unique, pred_counts = np.unique(label_pred, return_counts=True)
print(np.asarray((['Wake', 'N1', 'N2', 'N3', 'REM'], true_counts, pred_counts)).T)

