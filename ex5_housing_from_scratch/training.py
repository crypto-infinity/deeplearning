# %% [markdown]
# # Dependencies installation
# 
# Automatically installs all dependencies along with their imports.

# %%

import os
import pandas as pd
import numpy as np
import mlflow

mlflow.login()

import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras import layers, models # type: ignore

# %% [markdown]
# # Dataset pre-processing
# 
# Imports the datasets, while preprocessing and normalizing it.

# %%
url = "https://raw.githubusercontent.com/timothypesi/Data-Sets-For-Machine-Learning-/main/california_housing_train.csv"
dataset = pd.read_csv(url)

# %%
train_df, test_df = train_test_split(dataset, test_size=0.8, shuffle=True)

# %%
train_y = train_df.pop("median_house_value")
test_y = test_df.pop("median_house_value")

# %%
normalizer = tf.keras.layers.Normalization()

normalizer.adapt(train_df)
normalizer.adapt(test_df)

# %%
model = tf.keras.models.Sequential()

model.add(normalizer)
model.add(tf.keras.layers.Dense(200, activation="relu"))
model.add(tf.keras.layers.Dense(200, activation="relu"))
model.add(tf.keras.layers.Dense(200, activation="relu"))
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss=tf.keras.losses.MeanAbsoluteError()) #Watch out for the loss function

model.fit(train_df, train_y, epochs=300)

#test with other layers and with more epochs, and try adjusting the learning rate

# %%
model.evaluate(test_df, test_y)

# %%
prediction = model.predict(test_df)

test_df["predicted_median_house_value"] = prediction
test_df["median_house_value"] = test_y


