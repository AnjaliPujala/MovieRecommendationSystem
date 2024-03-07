
#importing dependencies

import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
  #converts data from text to numeric
from sklearn.metrics.pairwise import cosine_similarity

#reading the dataset

movie_data=pd.read_csv('movies_metadata.csv', low_memory=False)


#selecting features for recommendations

features_selected = ['genres', 'tagline', 'title' ]


#removing null values from the selected features

for feature in features_selected:
    movie_data[feature] = movie_data[feature].fillna('')

#combining the selected features

combined_features = movie_data['genres']+' '+movie_data['tagline']+' '+movie_data['title']


#transforming text data into numberic values or into vectors

vectorizer= TfidfVectorizer()
feature_vectors=vectorizer.fit_transform(combined_features)


#finding the similarity between the features
batch_size = 1000  # Adjust the batch size based on your available memory
num_samples = feature_vectors.shape[0]

# Initialize an empty similarity matrix
similarity_matrix = np.zeros((num_samples, num_samples))

# Compute cosine similarity in batches
for i in range(0, num_samples, batch_size):
    end_idx = min(i + batch_size, num_samples)
    similarity_matrix[i:end_idx, :] = cosine_similarity(feature_vectors[i:end_idx, :], feature_vectors)
print(similarity_matrix)

#taking movie name as input from the user

movie_name=input("Enter Movie name : ")

#listing of all movie name

list_of_movie_names = movie_data['title'].tolist()


#finding the similar movies

find_close_match = difflib.get_close_matches(movie_name,list_of_movie_names)
print("Movies that matches to your search: ",find_close_match)






