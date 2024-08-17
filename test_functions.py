# This file contains the test functions for the embedding_resnet Notebook
import pytest
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from runtime_embedding_resnet import load_embeddings_and_paths


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

    
import unittest
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Assuming calculate_histogram and find_similar_image functions are already defined

class TestFindSimilarImage(unittest.TestCase):

    def setUp(self):
        # Sample image and histograms for testing
        self.input_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # Mock histograms dictionary
        self.histograms = {
            "image1.jpg": np.random.rand(256, 256),
            "image2.jpg": np.random.rand(256, 256),
            "image3.jpg": np.random.rand(256, 256)
        }

        # Mock calculate_histogram function
        def mock_calculate_histogram(image, color_space):
            return np.random.rand(256, 256)

        # Replace the real calculate_histogram function with the mock
        global calculate_histogram
        calculate_histogram = mock_calculate_histogram

    def test_find_similar_image_single_neighbor(self):
        # Test with default single neighbor
        result = find_similar_image(self.input_image, self.histograms, n_neighbors=1)
        self.assertEqual(len(result), 1)
        self.assertIn(result[0][0], self.histograms.keys())
        self.assertIsInstance(result[0][1], float)

    def test_find_similar_image_multiple_neighbors(self):
        # Test with multiple neighbors
        n_neighbors = 2
        result = find_similar_image(self.input_image, self.histograms, n_neighbors=n_neighbors)
        self.assertEqual(len(result), n_neighbors)
        for r in result:
            self.assertIn(r[0], self.histograms.keys())
            self.assertIsInstance(r[1], float)

if __name__ == '__main__':
    unittest.main()
