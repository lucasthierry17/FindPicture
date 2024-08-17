
# necessary imports
import os
import sqlite3
from PIL import ImageCms
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import NearestNeighbors


icms = ImageCms


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


def display_images_from_paths(image_paths):
    """
    This function displays a list of images given their file paths.

    Input:
    - image_paths: A list of file paths to the images that should be displayed.

    Output:
    - The function displays the images in a matplotlib figure. It does not return any value.
    """
    plt.figure(figsize=(20, 5))
    for i, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.subplot(1, len(image_paths), i)
        plt.imshow(image_rgb)
        plt.title(f'Image {i}')
        plt.axis('off')
    plt.show()




