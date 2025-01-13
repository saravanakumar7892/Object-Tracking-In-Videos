import os
import random
from PIL import Image
from torchvision import transforms
import shutil

# Define directories
train_image_dir = 'E:/Object_detection/datasets/custom_dataset/train/images'
train_label_dir = 'E:/Object_detection/datasets/custom_dataset/train/labels'
augmented_image_dir = 'E:/Object_detection/datasets/custom_dataset/train/aug_images'
augmented_label_dir = 'E:/Object_detection/datasets/custom_dataset/train/aug_labels'

os.makedirs(augmented_image_dir, exist_ok=True)
os.makedirs(augmented_label_dir, exist_ok=True)

# Target count for balancing classes
target_count = 45612  # Class 3 instance count

# Transformation pipeline for augmentation
augmentation_pipeline = transforms.Compose([
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomResizedCrop(size=(416, 416), scale=(0.8, 1.0))
])

def augment_and_balance(image_dir, label_dir, output_image_dir, output_label_dir, target_count):
    """
    Augment images belonging to underrepresented classes to balance the dataset to a target class count.
    Args:
        image_dir: Path to images.
        label_dir: Path to labels.
        output_image_dir: Directory to save augmented images.
        output_label_dir: Directory to save augmented labels.
        target_count: The target count for underrepresented classes.
    """
    print(f"Starting augmentation process to match target count: {target_count}")
    total_files = len(os.listdir(label_dir))
    processed_files = 0

    for label_file in os.listdir(label_dir):
        processed_files += 1
        print(f"Processing file {processed_files}/{total_files}: {label_file}")

        label_path = os.path.join(label_dir, label_file)
        with open(label_path, 'r') as f:
            annotations = f.readlines()

        # Check if any class in the label is underrepresented
        augment_needed = False
        class_counts = {}

        for line in annotations:
            class_id = int(line.split()[0])
            if class_id not in class_counts:
                class_counts[class_id] = 0
            class_counts[class_id] += 1

        for class_id, count in class_counts.items():
            if count < target_count:
                augment_needed = True
                break

        if augment_needed:
            print(f"Augmentation needed for file: {label_file}")
            
            # Load the image
            image_file = os.path.splitext(label_file)[0] + '.jpg'
            image_path = os.path.join(image_dir, image_file)
            if os.path.exists(image_path):
                try:
                    image = Image.open(image_path)
                    print(f"Loaded image: {image_file}")
                except Exception as e:
                    print(f"Error loading image {image_file}: {e}")
                    continue

                # Perform augmentation
                for i in range(target_count - class_counts[class_id]):  # Augment based on the shortfall
                    augmented_image = augmentation_pipeline(image)

                    # Save augmented image
                    augmented_image_name = f"aug_{random.randint(1000, 9999)}_{image_file}"
                    augmented_image_path = os.path.join(output_image_dir, augmented_image_name)
                    augmented_image.save(augmented_image_path)

                    # Save corresponding label file
                    augmented_label_path = os.path.join(output_label_dir, os.path.splitext(augmented_image_name)[0] + '.txt')
                    shutil.copy(label_path, augmented_label_path)

                    print(f"Saved augmented image: {augmented_image_name} and label: {augmented_label_path}")
            else:
                print(f"Image not found: {image_file}")
        else:
            print(f"No augmentation needed for file: {label_file}")

    print("Augmentation process completed.")

# Run the augmentation process
augment_and_balance(
    train_image_dir,
    train_label_dir,
    augmented_image_dir,
    augmented_label_dir,
    target_count
)
