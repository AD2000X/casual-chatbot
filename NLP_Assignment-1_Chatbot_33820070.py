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

# Text Pre-processing
def clean_text(text):
    # lower case
    text = text.lower()
    # remove numbers
    text = re.sub(r'\d+', '', text)
    # remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # remove underline
    text = re.sub(r'\_', '', text)
    return text

# apply to DataFrame
dialogs_df['input_cleaned'] = dialogs_df['input'].apply(clean_text)
dialogs_df['output_cleaned'] = dialogs_df['output'].apply(clean_text)

# Text Pre-processing Visualisation

# word frequency before and after cleaning
words_before_clean = Counter(" ".join(dialogs_df["input"].dropna()).split())
words_after_clean = Counter(" ".join(dialogs_df["input_cleaned"].dropna()).split())

# convert to DataFrame to plot
df_before = pd.DataFrame(words_before_clean.most_common(10), columns=['Word', 'Frequency before cleaning'])
df_after = pd.DataFrame(words_after_clean.most_common(10), columns=['Word', 'Frequency after cleaning'])

# creat plot
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
sns.barplot(x='Frequency before cleaning', y='Word', data=df_before, palette='viridis')
plt.title('Top 10 Words before Cleaning')
plt.subplot(1, 2, 2)
sns.barplot(x='Frequency after cleaning', y='Word', data=df_after, palette='viridis')
plt.title('Top 10 Words after Cleaning')
plt.tight_layout()
plt.show()

# merge DataFrame columns into one string
text = " ".join(review for review in dialogs_df.input_cleaned.dropna())

# wordcloud
wordcloud = WordCloud(background_color="white").generate(text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# define a function to count letters in a string
def count_letters(text):
    return sum(c.isalpha() for c in text)

# apply to create new columns for the counts
dialogs_df['input_letters'] = dialogs_df['input'].apply(count_letters)
dialogs_df['output_letters'] = dialogs_df['output'].apply(count_letters)

# visualize the distribution of the number of letters
fig, ax = plt.subplots(1, 2, figsize=(20, 6), sharey=True)
sns.histplot(dialogs_df['input_letters'], bins=30, ax=ax[0], kde=True, color='blue')
ax[0].set_title('Distribution of Number of Letters in Input Texts')
ax[0].set_xlabel('Number of Letters')

sns.histplot(dialogs_df['output_letters'], bins=30, ax=ax[1], kde=True, color='green')
ax[1].set_title('Distribution of Number of Letters in Output Texts')
ax[1].set_xlabel('Number of Letters')

plt.tight_layout()
plt.show()

# Distribution of Number of words in each text

# calculate average word number
def count_words(text):
    return len(text.split())

dialogs_df['input_word_count'] = dialogs_df['input'].apply(count_words)
dialogs_df['output_word_count'] = dialogs_df['output'].apply(count_words)

# find entries with zero word count
zero_word_inputs = dialogs_df[dialogs_df['input_word_count'] == 0]
zero_word_outputs = dialogs_df[dialogs_df['output_word_count'] == 0]
print(zero_word_inputs[['input', 'input_cleaned']])
print(zero_word_outputs[['output', 'output_cleaned']])

# visualize
fig, axs = plt.subplots(1, 2, figsize=(20, 6), sharey=True)
sns.histplot(dialogs_df['input_word_count'], bins=30, ax=axs[0], kde=True, color="purple")
axs[0].set_title('Distribution of Number of Words in Input Texts')
axs[0].set_xlabel('Number of Words')
axs[0].set_ylabel('Frequency')

sns.histplot(dialogs_df['output_word_count'], bins=30, ax=axs[1], kde=True, color="orange")
axs[1].set_title('Distribution of Number of Words in Output Texts')
axs[1].set_xlabel('Number of Words')
# axs[1].set_ylabel('Frequency')  # share the y-axis label with the first subplot

plt.tight_layout()
plt.show()

# Visualize Average word length in each text

# calculate average word length
def average_word_length(text):
    words = text.split()
    word_lengths = [len(word) for word in words]
    return np.mean(word_lengths)

dialogs_df['input_avg_word_length'] = dialogs_df['input_cleaned'].apply(average_word_length)
dialogs_df['output_avg_word_length'] = dialogs_df['output_cleaned'].apply(average_word_length)

# visualize
fig, axs = plt.subplots(1, 2, figsize=(20, 6), sharey=True)
sns.histplot(dialogs_df['input_avg_word_length'], bins=30, ax=axs[0], kde=True, color="red")
axs[0].set_title('Distribution of Average Word Length in Input Texts')
axs[0].set_xlabel('Average Word Length')
axs[0].set_ylabel('Frequency')

sns.histplot(dialogs_df['output_avg_word_length'], bins=30, ax=axs[1], kde=True, color="cyan")
axs[1].set_title('Distribution of Average Word Length in Output Texts')
axs[1].set_xlabel('Average Word Length')
# axs[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# Words count in input and output text

# tokenize the cleaned texts into words
input_words = ' '.join(dialogs_df['input_cleaned']).split()
output_words = ' '.join(dialogs_df['output_cleaned']).split()

# count the occurrences of each word
input_word_counts = Counter(input_words)
output_word_counts = Counter(output_words)

# get the top N most common words
top_n = 20
top_input_words = input_word_counts.most_common(top_n)
top_output_words = output_word_counts.most_common(top_n)

# visualize the top words
fig, axs = plt.subplots(1, 2, figsize=(20, 6))

# input Texts
input_words, input_counts = zip(*top_input_words)
sns.barplot(x=list(input_words), y=list(input_counts), ax=axs[0], palette="Reds")
axs[0].set_title('Top Words in Input Texts')
axs[0].set_xlabel('Words')
axs[0].set_ylabel('Frequency')
axs[0].tick_params(axis='x', rotation=45)

# output Texts
output_words, output_counts = zip(*top_output_words)
sns.barplot(x=list(output_words), y=list(output_counts), ax=axs[1], palette="Greys")
axs[1].set_title('Top Words in Output Texts')
axs[1].set_xlabel('Words')
axs[1].set_ylabel('Frequency')
axs[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
