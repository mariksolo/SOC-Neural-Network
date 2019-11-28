# import shapefile
# import pyodbc
# #sf = shapefile.Reader("C:/Users/marik/Downloads/zinke_soil/zinke_soil/zinke_soil.dbf")
# #sf = shapefile.Reader("C:/Users/marik/Downloads/SOC_estimates_LatinAmerica/SOC_estimates_LatinAmerica/SOC_estimates_LatinAmerica.dbf")

# # !
# sf = shapefile.Reader("C:/Users/marik/Downloads/soils_CRA_ca_3761695_15/soils/cra_a_ca.dbf")
# print(sf.fields)

import csv
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)
oc = []
# with open('C:/Users/marik/OneDrive/Documents/Organic.csv') as csvDataFile:
#     csvReader = csv.reader(csvDataFile)
#     for row in csvReader:
#     	if row[-1] != '' and row[15] != '' and row[7] != '':
#         	oc.append([row[0], row[7], row[15], row[-1]])
column_names = ["n_tot", "ph_h2o", "oc"]
oc = pd.read_csv("C:/Users/marik/OneDrive/Documents/Organic.csv", usecols = column_names, skipinitialspace=True)
oc = oc.dropna()
# print(oc.head())

train_dataset = oc.sample(frac=0.8,random_state=0)
test_dataset = oc.drop(train_dataset.index)

train_stats = train_dataset.describe()
# print(train_stats['mean'])

train_labels = train_dataset.pop('oc')
test_labels = test_dataset.pop('oc')
print(train_dataset['n_tot'])
def norm(x):
  return (x - train_stats['oc']['mean']) / train_stats['std']
  # return x
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()

# example_batch = normed_train_data[:10]
# example_result = model.predict(example_batch)
# print(example_result)

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f}".format(mae))

