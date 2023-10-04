# Sentiment Analysis on Product Reviews (Machine Learning)

# Description

## Project Overview:

### Objective:

The objective of this project is to develop a machine learning model that can classify product reviews into positive or negative sentiments.

### Scope:

The analysis will focus on a specific category of products (e.g., electronics, books, movies) based on the available dataset. The project will involve preprocessing text data, training a sentiment analysis model, and evaluating its performance.

### Steps Involved:

1. **Problem Definition and Objectives**:
    - Define the main goal of the project: To build a model that can automatically classify product reviews as positive or negative sentiments.
2. **Data Collection**:
    - Acquire a labeled dataset of product reviews along with their corresponding sentiments. This dataset will serve as the foundation for training and testing the model.
3. **Data Preprocessing**:
    - Clean and prepare the text data. This includes removing special characters, converting text to lowercase, tokenization, and removing stop words.
4. **Data Exploration and Visualization**:
    - Analyze the dataset to understand its characteristics. Visualize the distribution of positive and negative reviews to check for class imbalances.
5. **Feature Extraction**:
    - Convert the preprocessed text data into numerical format using techniques like TF-IDF or word embeddings.
6. **Model Selection**:
    - Choose a machine learning algorithm suitable for sentiment analysis. Options include Logistic Regression, Support Vector Machines (SVM), Naive Bayes, or even deep learning models like LSTM or Transformer-based architectures.
7. **Model Training**:
    - Split the dataset into training and testing sets. Train the selected model on the training data.
8. **Model Evaluation**:
    - Use the testing set to evaluate the model's performance. Measure metrics such as accuracy, precision, recall, and F1-score to assess its effectiveness.

# Data Collection

The Yelp Reviews dataset is a popular choice for sentiment analysis projects. Here's how you can get started with the Yelp Reviews dataset:

### **Using the Yelp Reviews Dataset**

1. **Download the Dataset**:
    - You can download the Yelp Reviews dataset from **[this link](https://www.kaggle.com/yelp-dataset/yelp-dataset)**. Make sure you have a Kaggle account and are logged in.
2. **Unzip the Dataset**:
    - Once downloaded, unzip the file. You'll find several JSON files containing different types of data.
3. **Understand the Data**:
    - The dataset contains various files, but for sentiment analysis, you'll mainly be interested in the **`yelp_academic_dataset_review.json`** file. This file contains the reviews along with other information like user ID, business ID, etc.
4. **Load the Data**:
    - You can use Python to load and work with the dataset. Libraries like **`pandas`** are commonly used for data manipulation.

```python
import pandas as pd

# Load the dataset
df = pd.read_json('yelp_academic_dataset_review.json', lines=True)

# Check the first few rows
print(df.head())
```

## Data Preprocessing

### Cleaning the Data

- Remove any special characters, punctuation, and numbers from the reviews. This can be done using regular expressions or string manipulation.

```python
import re

def clean_text(text):
    # Remove special characters, numbers, and punctuation
    cleaned_text = re.sub('[^a-zA-Z\s]', '', text)
    return cleaned_text

# Apply the cleaning function to the 'text' column
df['cleaned_text'] = df['text'].apply(clean_text)
```

1. **Lowercasing**:
    - Convert all text to lowercase to ensure consistency.

```python
df['cleaned_text'] = df['cleaned_text'].str.lower()
```

1. **Tokenization**:
    - Split the text into individual words or tokens. This will make it easier to process.

```python
from nltk.tokenize import word_tokenize

df['tokenized_text'] = df['cleaned_text'].apply(word_tokenize)
```

1. **Removing Stopwords**:
    - Remove common words (e.g., "the", "is", "and") that don't provide much information.

```python
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

df['tokenized_text'] = df['tokenized_text'].apply(remove_stopwords)
```

# **Data Exploration and Visualization:**

1. **Basic Statistics**:
    - Get an overview of the dataset using descriptive statistics. This includes measures like mean, median, standard deviation, etc.

```python
# Descriptive statistics
print(df.describe())
```

```python
stars        useful         funny          cool                        date
count  10000.000000  10000.000000  10000.000000  10000.000000                       10000
mean       3.854300      0.889100      0.246500      0.335500  2015-04-17 08:27:40.820000
min        1.000000      0.000000      0.000000      0.000000         2005-03-01 17:47:15
25%        3.000000      0.000000      0.000000      0.000000  2013-11-14 11:16:35.500000
50%        4.000000      0.000000      0.000000      0.000000         2015-09-09 23:20:24
75%        5.000000      1.000000      0.000000      0.000000  2017-03-27 02:25:32.500000
max        5.000000     91.000000     26.000000     44.000000         2018-10-04 18:22:35
std        1.346719      2.092329      0.885221      1.051023                         NaN
```

1. **Class Distribution**:
    - Check the distribution of positive and negative sentiments in the dataset.

```python
import matplotlib.pyplot as plt

# Plot class distribution
plt.figure(figsize=(8, 6))
df['sentiment'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Class Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()
```

1. **Review Length Distribution**:
    - Analyze the distribution of review lengths. This can be useful for choosing appropriate sequence lengths when using models like LSTMs.

```python
# Calculate review lengths
df['review_length'] = df['processed_text'].apply(lambda x: len(x.split()))

# Plot review length distribution
plt.figure(figsize=(8, 6))
plt.hist(df['review_length'], bins=30, color='skyblue')
plt.title('Review Length Distribution')
plt.xlabel('Review Length')
plt.ylabel('Count')
plt.show()
```

1. **Top N Words**:
    - Identify the most frequently occurring words in the reviews.

```python
from collections import Counter

# Get the most common words
top_words = Counter(' '.join(df['processed_text']).split()).most_common(20)
top_words_df = pd.DataFrame(top_words, columns=['Word', 'Count'])

# Plot top words
plt.figure(figsize=(10, 6))
plt.barh(top_words_df['Word'], top_words_df['Count'], color='skyblue')
plt.title('Top 20 Words')
plt.xlabel('Count')
plt.ylabel('Word')
plt.gca().invert_yaxis()
plt.show()
```

# Feature Extraction

Feature extraction is the process of converting text data into a numerical format that can be fed into a machine learning model. There are several techniques you can use for feature extraction in NLP. Here, I'll cover two common methods: Bag-of-Words (BoW) and TF-IDF.

### **Bag-of-Words (BoW):**

BoW is a simple and effective method for feature extraction in NLP. It represents text as a set of words, ignoring grammar and word order.

```python
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the CountVectorizer
vectorizer = CountVectorizer(max_features=5000)  # You can adjust the 'max_features' parameter as needed

# Apply the vectorizer to the cleaned text
X_bow = vectorizer.fit_transform(df['cleaned_text'])

# Convert to array (optional)
X_bow = X_bow.toarray()

# X_bow now contains the feature vectors for your reviews
```

### **TF-IDF (Term Frequency-Inverse Document Frequency):**

TF-IDF is a more advanced technique that considers not only the frequency of a word in a document, but also the importance of the word in the entire corpus.

# Model Selection

Logistic Regression is performing well on our sentiment analysis task suggests that it's a good baseline model for this dataset. It's often surprising how well a simple model like Logistic Regression can perform in text classification tasks.

```python
from sklearn.linear_model import LogisticRegression

# Initialize the model
logistic_regression = LogisticRegression()

# Train the model
logistic_regression.fit(X_train, y_train)

# Evaluate the model
accuracy_lr = logistic_regression.score(X_test, y_test)
```

## Model Evaluation

```python
from sklearn.metrics import classification_report

# Predict on test set
y_pred = logistic_regression.predict(X_test)

# Generate classification report
print(classification_report(y_test, y_pred))
```

This will give you a detailed report including precision, recall, F1-score, and support for each class.

# **Conclusion:**

In this sentiment analysis project, we aimed to classify product reviews as either positive or negative using machine learning techniques. The dataset, sourced from Yelp, provided a diverse range of reviews across different businesses.

After a series of data preprocessing steps, including text cleaning, lowercasing, and tokenization, we explored two common feature extraction methods: Bag-of-Words (BoW) and TF-IDF. These techniques converted the text data into numerical format, which could be fed into machine learning models.

We experimented with several classification models, including Logistic Regression, Support Vector Machine (SVM), Naive Bayes, and Random Forest. Upon evaluation, it was found that Logistic Regression provided the best classification report, demonstrating its effectiveness as a baseline model for this task.

While Logistic Regression yielded satisfactory results, there is potential for further improvement through techniques such as hyperparameter tuning, feature engineering, and ensembling methods. Additionally, exploring more advanced models, such as neural networks, could lead to even higher performance.

In conclusion, this project demonstrates the successful application of machine learning techniques for sentiment analysis on product reviews. It serves as a solid foundation for future endeavors in natural language processing tasks.

# Author

[Arjith Praison](https://www.linkedin.com/in/arjith-praison-95b145184/)

University of Siegen
