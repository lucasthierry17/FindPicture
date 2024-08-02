# This file contains the test functions for the embedding_resnet Notebook
import pytest
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load embeddings and image paths from pickle files
def load_embeddings_and_paths(embeddings_path, image_paths_path):
    with open(embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)
    with open(image_paths_path, 'rb') as f:
        image_paths = pickle.load(f)
    return embeddings, image_paths
# Define the test function for pytest
def test_image_similarity():
    embeddings_path = 'embeddings.pkl'
    image_paths_path = 'image_paths.pkl'

    # Load the data
    embeddings, image_paths = load_embeddings_and_paths(embeddings_path, image_paths_path)

    # Check if we have at least two images to compare
    if len(embeddings) < 2:
        pytest.skip("Not enough embeddings to compare. Skipping test.")

    # Select the first two embeddings
    embedding1 = embeddings[0]
    embedding2 = embeddings[1]

    # Calculate similarity between the first two embeddings
    similarity_between_two_images = cosine_similarity([embedding1], [embedding2])[0][0]
    
    # Calculate similarity of each embedding with itself
    similarity_self1 = cosine_similarity([embedding1], [embedding1])[0][0]
    similarity_self2 = cosine_similarity([embedding2], [embedding2])[0][0]

    # Assert that the similarity of an image with itself is 1
    assert np.isclose(similarity_self1, 1.0), "Self-similarity for the first image is not 1."
    assert np.isclose(similarity_self2, 1.0), "Self-similarity for the second image is not 1."

    # Optionally, you can also assert the similarity between the two images if needed
    # For example, you can assert that it's not 1, or within some reasonable range, depending on your use case
    assert 0 <= similarity_between_two_images <= 1, "Similarity between the two images is out of bounds."

    # If everything passes, pytest will not show any output by default




