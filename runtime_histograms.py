import cv2
import matplotlib.pyplot as plt
import os
import sqlite3
from PIL import ImageCms
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from sklearn.neighbors import NearestNeighbors
import joblib
from sklearn.neighbors import NearestNeighbors


icms = ImageCms

# Connect to an existing one to store image metadata
database_name = "database_all_images.db"
root_folder = r"D:/data/image_data"

conn = sqlite3.connect(database_name)
c = conn.cursor()
conn.commit()


# ## Storing the storage path of every image in a variable
# This helps us to find and plot the similar images later in the code.

# In[3]:


# Fetch image paths from the database to work with the image files
# Execute the query to select all paths from the Images table
c.execute("SELECT Path FROM Images")

# Fetch all results
paths_tuples = c.fetchall()  # This will be a list of tuples

# Convert list of tuples to a list of strings
database_image_paths = [path[0] for path in paths_tuples]  # Extract the first element of each tuple


# In[14]:


# Calculate the histogram of an image using OpenCV
# The cv2.calcHist function computes the histogram for the given image
# In this context, histograms are used to compare images and determine similarity
def calculate_histogram(image, color_space):
    """
    This function computes the color histogram of an image in a specified color space, either HSV or RGB. 

    Input:
    - image: A single image represented as a NumPy array. This is the image whose histogram will be calculated.
    - color_space: A string indicating the color space to be used for the histogram calculation. It can be either "HSV" or "RGB"

    Output:
    - hist: A flattened, normalized histogram of the image, represented as a one-dimensional NumPy array.

    The image is converted from BGR (default OpenCV format) to HSV color space using cv2.cvtColor.
    Then, the histogram is calculated for the Hue and Saturation channels with 32 bins for each using cv2.calcHist. 
    The histogram is then normalized to ensure that the comparison between histograms is not biased by the size of the images.
    Finally, the normalized histogram is flattened into a one-dimensional array.
    
    """

    if color_space == "HSV":
        #The image is converted from BGR (default OpenCV format) to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])  
    elif color_space == "RGB":
        hist = cv2.calcHist([image], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256]) 
    else:
        pass

    hist = cv2.normalize(hist, hist).flatten()
    return hist



def process_image(image_path):
    """
    This function processes an image from a given path, calculates its histogram using the calculate_histogram function, and returns the path along with the histogram.

    Input: 
    - image_path: A string representing the file path of the image to be processed.

    Output:
    - A tuple with the image path and the corresponding histogram. If the image cannot be loaded, it returns 

    The image is read from the specified path using cv2.imread.
    If the image is successfully loaded (i.e., it is not None), the function calculates the histogram using the "HSV" color space.
    The function then returns a tuple containing the image path and the computed histogram.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is not None:
        # Getting the histogram for that image in HSV color space
        hist = calculate_histogram(image, "HSV")
        return (image_path, hist)
    return None




def load_histograms(save_path):
    """
    This function loads previously computed histograms from a pickle file if it exists. This allows the program to have access on every computed histogramm for every image.

    Input:
    - save_path: A string representing the file path where the histograms are saved.

    Output:
    - A dictionary where the keys are image paths and the values are their corresponding histograms.
    
    The function checks if the specified file exists.
    If the file exists, it loads the histograms using joblib.load (a method used for serializing Python objects).
    If the file does not exist, it returns an empty dictionary.

    """
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            histograms = joblib.load(f)
    else:
        histograms = {}
    return histograms




def compute_and_store_histograms(database_image_paths, save_path, save_interval=10000):
    """
    This function computes histograms for a list of images and periodically saves the results to a pickle file to avoid losing progress.

    Input:
    - database_image_paths: A list of strings, where each string is the file path of an image in the database.
    - save_path: A string representing the file path where histograms will be saved.
    - save_interval: An integer that specifies how many histograms to compute before saving intermediate results (default is 10,000).

    Output:
    - The function doesn’t return anything explicitly, but it saves the computed histograms to the specified file.

    The function first loads any existing histograms using load_histograms.
    It determines which images have already been processed and skips them.
    Using a ThreadPoolExecutor, it processes images in parallel to speed up the computation. It calculates histograms for batches of images and adds them to the existing dictionary of histograms.
    After processing each batch (of size save_interval), it saves the updated histogram dictionary to the specified save_path using joblib.dump.


    """
    histograms = load_histograms(save_path)
    processed_paths = set(histograms.keys())

    remaining_paths = [path for path in database_image_paths if path not in processed_paths]
    
    start_index = 0
    if processed_paths:
        start_index = database_image_paths.index(remaining_paths[0])
    
    with ThreadPoolExecutor() as executor:
        for i in range(start_index, len(database_image_paths), save_interval):
            batch_paths = database_image_paths[i:i+save_interval]
            results = list(tqdm(executor.map(process_image, batch_paths), total=len(batch_paths)))
            for result in results:
                if result is not None:
                    histograms[result[0]] = result[1]

            # Save intermediate results
            joblib.dump(histograms, save_path)
            print(f"Saved {len(histograms)} histograms to {save_path}")




def load_histograms_from_file(file_path):
    """
    This function is a simple wrapper to load histograms from a file.

    Input:
    - file_path: A string representing the file path from where histograms will be loaded.

    Output:
    - A dictionary of histograms where the keys are image paths and the values are the corresponding histograms
    """
    histograms = joblib.load(file_path)
    return histograms




def find_similar_image(input_image, histograms, n_neighbors=1):
    """
    This function finds images in the database that are most similar to the input image based on color histograms.

    Input:
    - input_image: The image for which we want to find similar images. This is a NumPy array representing the image.
    - histograms: A dictionary where keys are image paths and values are the corresponding histograms.
    - n_neighbors: An integer specifying the number of similar images to return (default is 1).

    Output:
    - A list of tuples where each tuple contains an image path and the corresponding distance to the input image. The list is ordered by similarity, with the most similar image first.

    The histogram of the input image is calculated using the calculate_histogram function in the "HSV" color space.
    The function extracts the paths and histogram data from the histograms dictionary and prepares them for similarity search.
    A nearest neighbors search is performed using NearestNeighbors from the scikit-learn library. The search uses Euclidean distance as the similarity metric.
    The function retrieves the image paths and distances of the most similar images.

    """
    input_hist = calculate_histogram(input_image, "HSV")

    # Prepare data for nearest neighbors search
    paths = list(histograms.keys())
    hist_data = np.array(list(histograms.values()))

    # Use KDTree or BallTree for efficient similarity search
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric='euclidean').fit(hist_data)
    distances, indices = nbrs.kneighbors([input_hist])

    # Retrieve the most similar images
    similar_images = [(paths[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
    return similar_images




def display_images(input_image_path, similar_images):
    """
    This function visually displays the input image alongside its most similar images found in the database.

    Input: 
    - input_image_path: A string representing the file path of the input image.
    - similar_images: A list of tuples, where each tuple contains an image path and the corresponding similarity distance.

    Output: 
    - The function does not return anything. It displays the images in a Matplotlib figure.

    The input image is loaded and converted from BGR to RGB color space.
    A plot is created with the input image on the left and the similar images on the right. Each similar image is labeled with its similarity distance.
    The function uses Matplotlib to display the images in a grid layout.
    
    """
    input_image = cv2.imread(input_image_path)
    input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(20, 5))
    
    # Plot input image
    plt.subplot(1, 6, 1)
    plt.imshow(input_image_rgb)
    plt.title('Input Image')
    plt.axis('off')
    
    # Plot similar images
    for i, (image_path, distance) in enumerate(similar_images, start=2):
        similar_image = cv2.imread(image_path)
        similar_image_rgb = cv2.cvtColor(similar_image, cv2.COLOR_BGR2RGB)
        plt.subplot(1, 6, i)
        plt.imshow(similar_image_rgb)
        plt.title(f'Similar {i-1}\nDist: {distance:.2f}')
        plt.axis('off')
    
    plt.show()
 # List of paths to database images
database_histogram_file = "histograms_all_images_0608.pkl"


# # Loading the histogram file and test the results

# In[5]:


histograms = load_histograms_from_file(database_histogram_file)


# ## Possible Results:
# 
# After running the code in the next cell, this could be your results.
# 
# ![alt text](output_example-1.png)

# In[7]:



input_image_path = "put an image here"
input_image = cv2.imread(input_image_path)
similar_images = find_similar_image(cv2.imread(input_image_path), histograms, n_neighbors=5)
# Request top 3 similar images
#display_images(input_image_path, similar_images)


# In[ ]:




