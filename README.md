# Find similar Image 

## Specifications
We have a database with around 450.000 images. Our goal is the find the closest image to an input image. We need to have three different ways to achieve our goal. 
Furthermore, we have the possibility to give 5 input images to our programm and it gives back 5 images, ideally related to all of the input images. 

Note: for a more detailed explanation of the code and our projct, please read our project documentation (Link_to_doku).

## How to run the code?
In order to run the code, you need to create your database with the images first. Therefore, you need to run the generator notebook and adjust the paths. Furhter instructions are in the code (Link_to_generator).
Secondly, you need choose the method you want to use, in order to find similar images. 
If you choose to use the embedding version, adjust the paths and run the following code: 
If you choose to use the color histogramm version, adjust the paths and run the following code: 
If you choose the combination of both methods, you first need to run the other two versions in order to use the combination. 

## Methods
### Method 1: Color histogramms
The first approach is based on color histogramms, more exact the HSV color space. 
You can find this approach in the Link_to_file
Results: 
![image](https://github.com/user-attachments/assets/14bd857a-4352-4ccf-b5e7-9d861ee4513b)

As you can see, we get 5 relatively similar images back for our input picture. The lower the similarity score is, the more related are two pictures. This is why Image 1 with a similarity score of 0.44 is considered the closest image. 

### Method 2: Embeddings
The second approach is based on embeddings. You can find this approach here: Link_to_file
Results: the following image shows the input picutre next to the 5 most similar images. This time a hihgher similarity score means that images are closer related to another. This is why Image 1 has the highest similarity with a score of 0.88.
![image](https://github.com/user-attachments/assets/d62584cc-d783-4d8d-90fb-0943c7eb4b86)


### Method 3: Combination of color histogramms and embeddings
Our third approach is based on a combination of the color histogramms and the embedding. 
Results: 


Here you can see the results for giving 5 images into the software, using the embeddings approach: 
These are the five input images...
![image](https://github.com/user-attachments/assets/e64b2b76-5410-4b54-8af3-ec43f1fb2e18)

And these are the five output images...
![image](https://github.com/user-attachments/assets/a71de536-5152-46fe-acc1-ef0242531f24)

### 2D-Plot of the pictures
The following picture shows the position of the images in x and y coordinates. Similar images are close to each other, meanwhile different images have a further images.
As you see, there are some folders, which have quite a lot of similar images, which results in a rather compressed area in the visualisation.
![image](https://github.com/user-attachments/assets/c7f13eeb-2279-4adc-8191-4f5ebdab5fa4)

