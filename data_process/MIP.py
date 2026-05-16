import cv2
import os
import numpy as np

#Define folder path
folder_path = "/ai/data/data/vessel/DIAS/test/images"

# Get all files in the folder
files = os.listdir(folder_path)

# Create a dictionary to store a list of images for each sequence ID
sequence_images = {}

# Iterate through each file in the folder
for file in files:
    # Split filename to get sequence ID and image ID
    sequence_id = file.split("_")[1]

    # read image
    image = cv2.imread(os.path.join(folder_path, file), cv2.IMREAD_GRAYSCALE)

    # If the sequence ID is not in the dictionary, add it
    if sequence_id not in sequence_images:
        sequence_images[sequence_id] = [image]
    else:
        sequence_images[sequence_id].append(image)

# Create a new folder to save the merged images
output_folder = "test_mip"
os.makedirs(output_folder, exist_ok=True)

# Merge images for each sequence ID and implement maximum density projection
for sequence_id, images in sequence_images.items():
    # Stack images together
    stacked_images = np.stack(images, axis=0)

    # Calculate maximum density projection
    max_density_projection = np.min(stacked_images, axis=0)
    # max_density_projection = np.where(max_density_projection > 100,255,0)

    #Save the maximum density projection image
    output_file = os.path.join(output_folder, f"{sequence_id}.jpg")
    cv2.imwrite(output_file, max_density_projection)

print("Merge and save completed.")