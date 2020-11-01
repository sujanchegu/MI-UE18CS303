import csv
import tensorflow as tf
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split



df = pd.read_csv('LBW_Dataset.csv')
df.Education.replace(np.NaN, 5, inplace=True)
df.Residence.fillna(df.Residence.mode()[0], inplace=True)
df["Delivery phase"].fillna(df["Delivery phase"].mode()[0], inplace=True)
df["Age"].fillna(df["Age"].median(axis=0), inplace=True)
df["Weight"].fillna(df["Weight"].mean(axis=0), inplace=True)
df["HB"].fillna(df["HB"].mean(axis=0), inplace=True)
df["BP"].fillna(df["BP"].mean(axis=0), inplace=True)
# print(df)


data = []


for i in range(len(df)):
    data.append({
            "evidence": [int(df.iloc[i,0]), int(df.iloc[i,1]), int(df.iloc[i,2]), int(df.iloc[i,3]), float(df.iloc[i,4]), int(df.iloc[i,5]), float(df.iloc[i, 6]), int(df.iloc[i,7]), int(df.iloc[i, 8])],
            "label": int(df.iloc[i,9])
        })



# Separate data into training and testing groups

evidence = [row["evidence"] for row in data]
labels = [row["label"] for row in data]
X_training, X_testing, y_training, y_testing = train_test_split(
    evidence, labels, test_size=0.3
)

# Create a neural network
model = tf.keras.models.Sequential([

    tf.keras.layers.Dense(16, input_shape=(9,)),
    tf.keras.layers.LeakyReLU(alpha=0.3),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(16, input_shape=(16,)),
    tf.keras.layers.LeakyReLU(alpha=0.3),
    tf.keras.layers.Dropout(0.5),



    tf.keras.layers.Dense(1, activation="softmax")
])

opt = tf.keras.optimizers.Adam(learning_rate=0.0006)

# Train neural network
model.compile(
    optimizer=opt,
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
model.fit(X_training, y_training, epochs=20)

# Evaluate how well model performs
model.evaluate(X_testing, y_testing, verbose=2)

