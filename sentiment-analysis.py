import pandas as pd
import re
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the dataset
df = pd.read_json('yelp_academic_dataset_review.json', lines=True, nrows=10000)

# Check the first few rows
print(df.head())


def clean_text(text):
    # Remove special characters, numbers, and punctuation
    cleaned_text = re.sub('[^a-zA-Z\s]', '', text)
    return cleaned_text

# Apply the cleaning function to the 'text' column
df['cleaned_text'] = df['text'].apply(clean_text)

df['cleaned_text'] = df['cleaned_text'].str.lower()

# Descriptive statistics
print(df.describe())

# Plot class distribution
plt.figure(figsize=(8, 6))
df['stars'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title('Class Distribution')
plt.xlabel('Stars')
plt.ylabel('Count')
plt.show()

# Calculate review lengths
df['review_length'] = df['cleaned_text'].apply(lambda x: len(x.split()))

# Plot review length distribution
plt.figure(figsize=(8, 6))
plt.hist(df['review_length'], bins=30, color='skyblue')
plt.title('Review Length Distribution')
plt.xlabel('Review Length')
plt.ylabel('Count')
plt.show()

# Get the most common words
top_words = Counter(' '.join(df['cleaned_text']).split()).most_common(20)
top_words_df = pd.DataFrame(top_words, columns=['Word', 'Count'])

# Plot top words
plt.figure(figsize=(10, 6))
plt.barh(top_words_df['Word'], top_words_df['Count'], color='skyblue')
plt.title('Top 20 Words')
plt.xlabel('Count')
plt.ylabel('Word')
plt.gca().invert_yaxis()
plt.show()

# Initialize the CountVectorizer
vectorizer = CountVectorizer(max_features=5000)  # You can adjust the 'max_features' parameter as needed

# Apply the vectorizer to the cleaned text
X_bow = vectorizer.fit_transform(df['cleaned_text'])

# Convert to array (optional)
X_bow = X_bow.toarray()

# X_bow now contains the feature vectors for your reviews


# Initialize the TfidfVectorizer
vectorizer_tfidf = TfidfVectorizer(max_features=5000)  # You can adjust the 'max_features' parameter as needed

# Apply the vectorizer to the cleaned text
X_tfidf = vectorizer_tfidf.fit_transform(df['cleaned_text'])

# Convert to array (optional)
X_tfidf = X_tfidf.toarray()

# X_tfidf now contains the feature vectors for your reviews


X_train, X_test, y_train, y_test = train_test_split(X_bow, df['stars'], test_size=0.2, random_state=42)



# Initialize the model
logistic_regression = LogisticRegression()

# Train the model
logistic_regression.fit(X_train, y_train)

# Evaluate the model
accuracy_lr = logistic_regression.score(X_test, y_test)

# Predict on test set
y_pred = logistic_regression.predict(X_test)

# Generate classification report
print(classification_report(y_test, y_pred))