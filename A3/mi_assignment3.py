# -*- coding: utf-8 -*-
"""MI_Assignment3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GUhlSeywM3EgjvzoxW4zwwX2I-Aq6Ngo
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# from google.colab import files
# uploaded = files.upload()

import os
print(os.getcwd())
print(os.listdir())

df = pd.read_csv('LBW_Dataset.csv')

df

df.isnull().sum(axis = 0)

#Analyse each column and checkout diff things from here

#Education - All 5 - Scam
df.Education.replace(np.NaN, 5, inplace=True)
df.Education.unique()

#Residence - use mode to replace it
df.Residence.fillna(df.Residence.mode()[0], inplace=True)
df.Residence.unique()

#Delivery Phase - use mode to replace
df["Delivery phase"].fillna(df["Delivery phase"].mode()[0], inplace=True)
df["Delivery phase"].unique()

df.isnull().sum(axis=0)

# Age - use median to replace
df["Age"].fillna(df["Age"].median(axis=0), inplace=True)
df["Age"].unique()

# Weight - use mean to replace
df["Weight"].fillna(df["Weight"].mean(axis=0), inplace=True)
df["Weight"].unique()

# HB - use mean to replace
df["HB"].fillna(df["HB"].mean(axis=0), inplace=True)
df["HB"].unique()

# BP - use mean to replace
df["BP"].fillna(df["BP"].mean(axis=0), inplace=True)
df["BP"].unique()

df.isnull().sum(axis=0)
