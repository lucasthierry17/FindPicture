#!/usr/bin/env python
# coding: utf-8

# In[1]:


# necessary imports
import os
import sqlite3
from PIL import Image, ImageCms
from tqdm import tqdm
import numpy as np




icms = ImageCms


# In[2]:


# creating a database using sql lite
database_name = "database_all_images.db"
root_folder = r"D:/data/image_data"

conn = sqlite3.connect(database_name)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS Images
                 (ID INTEGER PRIMARY KEY,
                 Name TEXT,
                 Path TEXT,
                 Size INTEGER
                 )''')
conn.commit()



# # Diesen Code Abschnitt lösen sobald die Datenbank erstellt wurde um keine Verwirrung zu stiften

# In[12]:


def traverse_folders(root_folder):
    for root, dirs, files in os.walk(root_folder):
        total_files = len(files)
        for i, file in enumerate(files, 1):
            if file.endswith(('.jpg', '.jpeg', '.png', '.tiff')):  # list all the formats
                file_path = os.path.join(root, file)  # get file path
                try:
                    file_size = os.path.getsize(file_path)  # get file size
                    image = Image.open(file_path)
                    yield (file, file_path, file_size), i / total_files * 100
                except Exception as e:
                    pass
                    # Catch any other exceptions and log if needed
                    # print(f"Unexpected error with file {file_path}: {e}")
            else:
                pass

# Function to insert data into the database
def insert_into_database(database_name, data):
    conn = sqlite3.connect(database_name)
    c = conn.cursor()
    c.execute("INSERT INTO Images (Name, Path, Size) VALUES (?, ?, ?)", data)
    conn.commit()
    
for data, progress in tqdm(traverse_folders(root_folder), total=500000, desc="Processing images", unit="%"):
    insert_into_database(database_name, data)

# print("Database creation and data insertion completed.")

