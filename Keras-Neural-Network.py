import os
import glob
import csv
import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

import tensorflow as tf


data_path = '/Users/johakim/Projects/Lab-Project/Feature-Extracted'
os.chdir(data_path)
data_files = glob.glob('*.csv')

data_df_collection = []
for file in data_files:
    data_df = pd.read_csv(file)
    data_df_collection.append(data_df)

# Define per-fold score containers
num_folds = len(data_df_collection)
loss_per_fold = []

for fold in range(num_folds):
    # choose one file for testing and rest for training
    test_set = data_df_collection[fold]
    train_set = pd.concat([y for i, y in enumerate(data_df_collection) if i != fold], ignore_index=True)

    test_X = test_set.iloc[:, :-2]
    test_y = test_set['Gait Percent']

    train_X = train_set.iloc[:, :-2]
    train_y = train_set['Gait Percent']

    # create model
    model = Sequential()
    model.add(Dense(20, activation='elu', input_shape=(train_X.shape[1],)))
    model.add(Dense(20, activation='elu'))
    model.add(Dense(20, activation='elu'))
    model.add(Dense(1))

    # compile model
    model.compile(loss='mean_squared_error', optimizer='adam')

    # fit model
    print('-'*80)
    print(f'Training for fold {fold}...')
    epochs = 100
    # early_stopping_monitor = EarlyStopping(patience=5)
    history = model.fit(train_X, train_y, validation_split=0.2, epochs=epochs, verbose=0)

    # learning curve
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.figure()
    plt.plot(train_loss, 'bo', label='Training')
    plt.plot(val_loss, 'r', label='Validation')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.title('Learning Curve')
    plt.legend(['Training', 'Validation'], loc='upper left')
    path = '/Users/johakim/Projects/Lab-Project/Learning-Curves/fold_' + str(fold) + '.png'
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    # plt.show()

    # generate generalization metrics
    loss = model.evaluate(test_X, test_y, verbose=0)
    print('Test loss:', loss)
    loss_per_fold.append(loss)

# generate average scores
print('-'*80)
print('Average loss for all folds:')
avg_loss = np.mean(loss_per_fold)
print(f'> Loss: {avg_loss}')

model_details = [(20, 'elu'), (20, 'elu'), (20, 'elu'), avg_loss]
results_fname = '/Users/johakim/Projects/Lab-Project/Model-Losses.csv'
if os.path.isfile(results_fname):
    with open(results_fname, 'a') as fd:
        writer = csv.writer(fd)
        writer.writerow(model_details)
else:
    results = [model_details]
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_fname, index=False, header=False)



