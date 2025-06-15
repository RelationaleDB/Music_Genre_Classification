#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import scipy

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, accuracy_score 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[2]:


# Load the dataset
df = pd.read_csv('music_dataset_mod.csv')  


# In[3]:


# Create a copy for safe experimentation
df_copy = df.copy()


# In[4]:


# Peek at the first few rows
print(df_copy.head())


# In[6]:


# Basic info about the DataFrame
df_copy.info()


# In[7]:


# Check for missing values in each column
print(df_copy.isnull().sum())


# In[8]:


# Get unique genres
unique_genres = df_copy['Genre'].unique()
print(unique_genres)


# In[9]:


# Optional: Count them
print(f"Number of unique genres: {len(unique_genres)}")


# In[10]:


# Set the plot size
plt.figure(figsize=(12, 6))

# Create a count plot for Genre
sns.countplot(data=df_copy, x='Genre', order=df_copy['Genre'].value_counts().index)

# Add title and labels
plt.title('Distribution of Music Genres')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.xticks(rotation=45)  # Rotate for better readability
plt.tight_layout()
plt.show()


# In[11]:


# Create a new copy excluding rows with missing Genre
df_cleaned = df_copy.dropna(subset=['Genre'])


# In[12]:


# Check shape to see how many rows were removed
print(f"Original shape: {df_copy.shape}")
print(f"Cleaned shape: {df_cleaned.shape}")


# In[13]:


# Separate the target and features
X = df_cleaned.drop(columns=['Genre'])  # all columns except Genre
y = df_cleaned['Genre']  # target column


# In[14]:


# Initialize encoder and apply transformation
le = LabelEncoder()
y_encoded = le.fit_transform(y)


# In[17]:


# Show a few results
print("Original genres:", list(y[:7]))
print("Encoded genres:", y_encoded[:7])


# In[18]:


# Create a new DataFrame including the encoded target
df_encoded = df_cleaned.copy()
df_encoded['Genre'] = y_encoded


# In[19]:


# Compute correlation matrix
corr_matrix = df_encoded.corr()

# Set plot size
plt.figure(figsize=(12, 10))

# Draw heatmap
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)

plt.title('Correlation Matrix of Music Features and Encoded Genre')
plt.tight_layout()
plt.show()


# In[20]:


# Drop target column and standardize feature data
features = df_encoded.drop(columns=['Genre'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)


# In[21]:


# Create PCA instance
pca = PCA()

# Fit and transform the scaled data
X_pca = pca.fit_transform(X_scaled)


# In[22]:


# Explained variance by each component
explained_variance = pca.explained_variance_ratio_

# Show first few components
for i, var in enumerate(explained_variance[:10]):
    print(f"PC{i+1}: {var:.4f} variance explained")


# In[23]:


# Cumulative variance
cumulative_variance = np.cumsum(explained_variance)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o', linestyle='--')

plt.title('Cumulative Explained Variance by PCA Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.axhline(y=0.80, color='r', linestyle='--', label='80% Variance Threshold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[24]:


# Find number of components needed for 80% variance
num_components = np.argmax(cumulative_variance >= 0.80) + 1
print(f"Number of components to retain ≥80% variance: {num_components}")


# In[25]:


# Reapply PCA with chosen number of components
pca_final = PCA(n_components=num_components)
X_reduced = pca_final.fit_transform(X_scaled)

# Optionally convert to DataFrame
X_pca_df = pd.DataFrame(X_reduced, columns=[f'PC{i+1}' for i in range(num_components)])
print(X_pca_df.head())


# In[27]:


# Split PCA-transformed data
X_pca_train, X_pca_test, y_train, y_test = train_test_split(
    X_reduced, y_encoded, test_size=0.3, random_state=42
)


# In[28]:


# Create and train the model
model_pca = LogisticRegression(max_iter=10000)
model_pca.fit(X_pca_train, y_train)


# In[42]:


# Predict and evaluate
y_pca_pred = model_pca.predict(X_pca_test)

# Accuracy
acc_pca = accuracy_score(y_test, y_pca_pred)
print(f"PCA Model Accuracy: {acc_pca:.4f}")

# Detailed report
print("Classification Report (PCA):")
print(classification_report(y_test, y_pca_pred))


# In[30]:


# Split original scaled data
X_orig_train, X_orig_test, y_train_orig, y_test_orig = train_test_split(
    X_scaled, y_encoded, test_size=0.3, random_state=42
)

# Train logistic regression on original data
model_orig = LogisticRegression(max_iter=10000)
model_orig.fit(X_orig_train, y_train_orig)

# Predict and evaluate
y_orig_pred = model_orig.predict(X_orig_test)

# Accuracy
acc_orig = accuracy_score(y_test_orig, y_orig_pred)
print(f"Original Model Accuracy: {acc_orig:.4f}")

# Detailed report
print("Classification Report (Original Data):")
print(classification_report(y_test_orig, y_orig_pred))


# In[31]:


# Find rows where Genre is missing
unknown_genre_df = df[df['Genre'].isnull()].copy()

# Preview
print(f"Number of unknown genre tracks: {len(unknown_genre_df)}")
unknown_genre_df.head()


# In[32]:


# Drop Genre column
X_unknown = unknown_genre_df.drop(columns=['Genre'])

# Apply same scaler (DO NOT re-fit!)
X_unknown_scaled = scaler.transform(X_unknown)

# If using PCA-based model, apply PCA transform
X_unknown_transformed = pca_final.transform(X_unknown_scaled)


# In[33]:


# Predict using the PCA-based model
unknown_genre_predictions = model_pca.predict(X_unknown_transformed)


# In[34]:


# Convert numeric labels back to original genres
predicted_genre_labels = le.inverse_transform(unknown_genre_predictions)

# Show a few
predicted_genre_labels[:5]


# In[35]:


# Make a copy to avoid changing original directly
df_filled = df.copy()

# Assign predicted genres
df_filled.loc[df_filled['Genre'].isnull(), 'Genre'] = predicted_genre_labels


# In[36]:


# Confirm no missing genres remain
print("Missing genres after filling:", df_filled['Genre'].isnull().sum())

# Show updated rows that were previously missing
df_filled.loc[df['Genre'].isnull()].head()


# In[ ]:




