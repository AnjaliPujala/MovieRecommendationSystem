# MovieRecommendationSystem

This project implements a simple movie recommendation system using content-based filtering. It leverages the TfidfVectorizer and cosine similarity to recommend movies based on their genres, tagline, and title.

## Importing Dependencies

- NumPy
- Pandas
- difflib
- Scikit-learn

## Dataset

The dataset is read from the 'movies_metadata.csv' file. Ensure that the actual dataset filename is used if different.

## Feature Selection

The following features are selected for movie recommendations:

- Genres
- Tagline
- Title

## Data Preprocessing

Null values in the selected features are handled by filling them with an empty string.

## Text Vectorization

The selected features are combined, and TfidfVectorizer is used to transform the text data into numeric values or vectors.

## Similarity Computation

Cosine similarity is computed between the feature vectors in batches to build a similarity matrix.

```python
# Code snippet for similarity computation
# ...

# Compute cosine similarity in batches
for i in range(0, num_samples, batch_size):
    end_idx = min(i + batch_size, num_samples)
    similarity_matrix[i:end_idx, :] = cosine_similarity(feature_vectors[i:end_idx, :], feature_vectors)
print(similarity_matrix)
