import sys
import os
import pandas as pd

import re
import numpy
from collections import Counter

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# data split and feature engineering
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.preprocessing import LabelEncoder

# Machine Learning
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import joblib

# deploy chatter bot
from flask import Flask, request, jsonify


# EDA (Exploratory Data Analysis)
current_dir = os.getcwd()

file_name_dialogs = 'dialogs.csv'
file_path_dialogs = os.path.join(current_dir, file_name_dialogs)

with open(file_path_dialogs, 'r', encoding='utf-8') as file:
    dialogs = file.readlines()

total_dialogs = len(dialogs)

# print out some basic information to understand the data
total_dialogs, dialogs[:20]

# DataFrame shape
dialogs_df = dialogs_df.shape
dialogs_df

# path of the current directory, file name and file path
current_dir = os.getcwd()
file_name_dialogs = 'dialogs.csv'
file_path_dialogs = os.path.join(current_dir, file_name_dialogs)
dialogs_df = pd.read_csv(file_path_dialogs)

# load DataFrame
dialogs_df['input tokens'] = dialogs_df['input'].apply(lambda x: len(x.split()))
dialogs_df['output tokens'] = dialogs_df['output'].apply(lambda x: len(x.split()))

# setting the style for plots
plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
sns.set_palette('Set2')

# histograms for the tokens in questions and answers
sns.histplot(x='input tokens', data=dialogs_df, kde=True, ax=ax[0])
sns.histplot(x='output tokens', data=dialogs_df, kde=True, ax=ax[1])

# joint kernel density estimate plot
sns.jointplot(x='input tokens', y='output tokens', data=dialogs_df, kind='kde', fill=True, cmap='YlGnBu')

plt.show()

# load the data
dialogs = os.path.join(current_dir, file_name_dialogs)

# basic statistics
print("Basic Information:")
print(dialogs_df.describe(include='all'))
print("\nUnique values per column:")
print(dialogs_df.nunique())

# intent quantity
intent_counts = dialogs_df['intent'].value_counts()

# new DataFrame contains 'intent' and its quantity
intent_counts_df = pd.DataFrame({'Intent': intent_counts.index, 'Count': intent_counts.values})

# plotchat
plt.figure(figsize=(10, 6))
sns.countplot(data=dialogs_df, y='intent', order=intent_counts.index)
plt.title('Distribution of Intents')
plt.xlabel('Count')
plt.ylabel('Intent')

# tag number
for i, count in enumerate(intent_counts.values):
    plt.text(count + 5, i, str(count), ha='center', va='center')

plt.show()

# common inputs
common_inputs = dialogs_df['input'].value_counts().head(10)
print("\nTop 10 common inputs:")
print(common_inputs)

# common outputs
common_outputs = dialogs_df['output'].value_counts().head(10)
print("\nTop 10 common outputs:")
print(common_outputs)

