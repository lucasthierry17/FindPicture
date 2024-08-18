import unittest
import numpy as np
from unittest.mock import Mock
from runtime_embedding_resnet_final import find_similar_images
import sqlite3

class TestFindSimilarImages(unittest.TestCase):

    def setUp(self):
        # Create a mock model that returns a predefined embedding for the input image
        self.mock_model = Mock()
        self.mock_model.predict.return_value = np.array([[0.5, 0.5, 0.5, 0.5]])

        # Create mock embeddings for the database images
        self.embeddings = np.array([
            [0.1, 0.1, 0.1, 0.1],
            [0.5, 0.5, 0.5, 0.5],
            [0.9, 0.9, 0.9, 0.9]
        ])

        # Create a list of image paths corresponding to the embeddings
        self.image_paths = [
            "image_1.jpg",
            "image_2.jpg",
            "image_3.jpg"
        ]

    def test_find_similar_images(self):
        # Path to the input image (irrelevant here since we're mocking the model)
        input_img_path = "test_image.jpg"

        # Mock extract_embedding to return the same embedding as the mock model
        with unittest.mock.patch('runtime_embedding_resnet_final.extract_embedding', return_value=np.array([0.5, 0.5, 0.5, 0.5])):
            # Call the function with mock data
            result = find_similar_images(input_img_path, self.mock_model, self.embeddings, self.image_paths, top_n=2)

        # Print the result for debugging
        print(f"Result: {result}")

        # Expected normalized distances might not match exactly but should be close to:
        # 0.0 for the most similar, and the next closest should be significantly higher.
        self.assertEqual(result[0][0], "image_2.jpg")  # Most similar image
        self.assertEqual(result[1][0], "image_1.jpg")  # Next most similar image

        # Since we expect some variation in the normalization, let's use `assertAlmostEqual`
        self.assertAlmostEqual(result[0][1], 0.0, delta=0.1)  # Should be very close to 0
        self.assertAlmostEqual(result[1][1], 1.0, delta=0.5)  # Should be close to 1 but can vary

    def test_no_similar_images(self):
        # Test the function with an image that has no close match in the database
        input_img_path = "test_image.jpg"

        # Mock extract_embedding to return an embedding not close to any in the database
        with unittest.mock.patch('runtime_embedding_resnet_final.extract_embedding', return_value=np.array([0.0, 0.0, 0.0, 0.0])):
            result = find_similar_images(input_img_path, self.mock_model, self.embeddings, self.image_paths, top_n=2)

        # Print the result for debugging
        print(f"Result: {result}")

        # Expect distances to be reasonably high since the input embedding is far from others
        self.assertEqual(len(result), 2)
        for r in result:
            self.assertTrue(r[1] >= 0.1, f"Distance too low: {r[1]}")  # Expecting normalized distances above 0.1

class TestSQLiteConnection(unittest.TestCase):

    def test_sqlite_connection(self):
        db_path = 'database_all_images.db'  # Hier den Pfad zur Datenbank anpassen

        try:
            # Stelle eine Verbindung zur SQLite-Datenbank her
            connection = sqlite3.connect(db_path)

            # Erstelle ein Cursor-Objekt
            cursor = connection.cursor()

            # Führe eine einfache SQL-Abfrage aus
            cursor.execute("SELECT 1")

            # Überprüfe das Ergebnis
            result = cursor.fetchone()
            self.assertEqual(result[0], 1, "Die Verbindung zur SQLite-Datenbank ist fehlgeschlagen.")
            print("SQLite-Verbindung funktioniert einwandfrei.")

        except sqlite3.Error as e:
            self.fail(f"SQLite-Fehler: {e}")
        finally:
            if connection:
                connection.close()

if __name__ == '__main__':
    unittest.main()
