import os

# Set the paths for the dataset
train_image_path = "E:/Object_detection/datasets/custom_dataset/train/aug_images"
train_label_path = "E:/Object_detection/datasets/custom_dataset/train/aug_labels"

# Get the list of image files and label files
train_images = [f for f in os.listdir(train_image_path) if f.endswith(('.jpg', '.png'))]
train_labels = [f for f in os.listdir(train_label_path) if f.endswith('.txt')]

print(f"Number of training images: {len(train_images)}")
print(f"Number of training labels: {len(train_labels)}")
