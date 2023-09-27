"""import cv2
import numpy as np
import os
import glob
import mahotas as mt
from sklearn.svm import LinearSVC
from tkinter import Tk
from tkinter.filedialog import askdirectory, askopenfile

# function to extract haralick textures from an image
def extract_features(image):
	# calculate haralick texture features for 4 types of adjacency
	textures = mt.features.haralick(image)

	# take the mean of it and return it
	ht_mean  = textures.mean(axis=0)
	return ht_mean

# load the training dataset
train_path = askdirectory(title = "Select a Folder")
train_names = os.listdir(train_path)

# empty list to hold feature vectors and train labels
train_features = []
train_labels   = []

# loop over the training dataset
print ("[STATUS] Started extracting haralick textures..")
for train_name in train_names:
	cur_path = train_path + "/" + train_name
	cur_label = train_name
	i = 1

	for file in glob.glob(cur_path + "/*.tif"):
		print ("Processing Image - {} in {}".format(i, cur_label))
		# read the training image
		image = cv2.imread(file)

		# convert the image to grayscale
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# extract haralick texture from the image
		features = extract_features(gray)

		# append the feature vector and label
		train_features.append(features)
		train_labels.append(cur_label)

		# show loop update
		i += 1

# have a look at the size of our feature vector and labels
print ("Training features: {}".format(np.array(train_features).shape))
print ("Training labels: {}".format(np.array(train_labels).shape))

# create the classifier
print ("[STATUS] Creating the classifier..")
clf_svm = LinearSVC(random_state=9)

# fit the training data and labels
print ("[STATUS] Fitting data/label to model..")
clf_svm.fit(train_features, train_labels)

# loop over the test images
test_path = askdirectory(title = "Select a Folder")
for file in glob.glob(test_path + "/*.tif"):
	# read the input image
	image = cv2.imread(file)

	# convert to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# extract haralick texture from the image
	features = extract_features(gray)

	# evaluate the model and predict label
	prediction = clf_svm.predict(features.reshape(1, -1))[0]

	# show the label
	cv2.putText(image, prediction, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
	print ("Prediction - {}".format(prediction))

	# display the output image
	cv2.imshow("Test_Image", image)
	cv2.waitKey(0)"""

"""	# importing required libraries
import numpy as np
import mahotas
import os
import tkinter as TK
from tkinter.filedialog import askdirectory, askopenfile
from pylab import imshow, show
  
# loading image
train_path = askdirectory(title = "Select a Folder")
train_names = os.listdir(train_path)
for names in train_names:
	img = mahotas.imread(train_path + "/" + names)
    
	# filtering the image
	img = img[:, :, 0]
     
	# setting gaussian filter
	gaussian = mahotas.gaussian_filter(img, 15)
  
	# setting threshold value
	gaussian = (gaussian > gaussian.mean())
  
	# making is labelled image
	labelled, n = mahotas.label(gaussian)
 
	# showing image
	print("Labelled Image")
	imshow(labelled)
	show()
 
	# getting haralick features
	h_feature = mahotas.features.haralick(labelled)
 
	# showing the feature
	print("Haralick Features")
	imshow(h_feature)
	show()"""

import os
import numpy as np
import mahotas.features.texture as texture
import csv
import tkinter as TK
from tkinter.filedialog import askdirectory, askopenfile
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray

# Function to extract Haralick texture features from an image
def extract_haralick_features(image):
    # Convert the image to grayscale
    image_gray = image.convert("L")
    
    # Convert the image to a NumPy array
    image_array = np.array(image_gray)
    
    # Extract Haralick texture features
    haralick_features = texture.haralick(image_array)
    
    # Flatten the 2D array into a 1D array
    haralick_features = haralick_features.flatten()
    
    return haralick_features

def extract_color_haralick_features(image):
    # Convert image to grayscale
    gray_image = rgb2gray(image)

    # Normalize the grayscale image to [0, 255]
    gray_image = (gray_image * 255).astype(np.uint8)

    # Compute the color co-occurrence matrix
    distances = [1]  # Define the distance between pixels for co-occurrence
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Define the angles for co-occurrence
    properties = ['contrast', 'correlation', 'energy', 'homogeneity']
    color_cooccurrence_matrix = graycomatrix(gray_image, distances, angles, levels=256, symmetric=True, normed=True)

    # Extract color Haralick features
    features = []
    for prop in properties:
        feature = graycoprops(color_cooccurrence_matrix, prop)
        features.append(feature.flatten())

    # Concatenate the extracted features
    features = np.concatenate(features)

    return features

# Set the path to the folder containing the image tiles
folder_path = askdirectory(title = "Select a Folder")

# Set List with all the names of the tiles in the Path Folder
folder_list = os.listdir(folder_path)

# Set the output CSV file path
output_path = askdirectory(title = "Select a Folder")
output_file = output_path + "/" + "RGB-haralick_features.csv"

# Create a list to store the feature vectors
feature_vectors = []

# Iterate through each image tile in the folder
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    
    # Load the image
    image = Image.open(file_path)
    
    # Extract Haralick texture features
    haralick_features = extract_color_haralick_features(image)
    
    # Append the features to the list
    feature_vectors.append(haralick_features)

# Write the feature vectors to a CSV file
with open(output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    
    # Write the header row
    writer.writerow(["Tile"] + [f"Feature_{i+1}" for i in range(len(feature_vectors[0]))])
    
    # Write the feature vectors
    for i, features in enumerate(feature_vectors):
        writer.writerow([folder_list[i]] + list(features))

print("Haralick texture features extracted and saved to", output_file)
