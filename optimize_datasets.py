import os
import random
import shutil  
from PIL import Image

# Paths to datasets
DATASETS = {
    "coco2014": {
        "base_dir": "E:/Object_detection/datasets/coco2014",
        "yaml_file": "E:/Object_detection/datasets/coco2014/coco2014_optimized.yaml",
        "class_names": ["class1", "class2", "class3"]  # Replace with actual COCO class names
    },
    "PASCAL_VOC": {
        "base_dir": "E:/Object_detection/datasets/PASCAL_VOC",
        "yaml_file": "E:/Object_detection/datasets/PASCAL_VOC/pascal_voc_optimized.yaml",
        "class_names": ["class1", "class2"]  # Replace with actual Pascal VOC class names
    },
    "custom_dataset": {
        "base_dir": "E:/Object_detection/datasets/custom_dataset",
        "yaml_file": "E:/Object_detection/datasets/custom_dataset/custom_dataset_optimized.yaml",
        "class_names": ["custom1", "custom2"]  # Replace with your custom class names
    }
}

# Resize and compression parameters
RESIZE_SIZE = (640, 640)  # Image resize dimensions
COMPRESSION_QUALITY = 85  # JPEG compression quality (1-100)
SAMPLE_FRACTION = 0.2  # Fraction of the dataset to keep (20%)

def optimize_dataset(dataset_name, dataset_info):
    print(f"Processing dataset: {dataset_name}")
    base_dir = dataset_info["base_dir"]
    yaml_file = dataset_info["yaml_file"]
    class_names = dataset_info["class_names"]

    # Directories for train, val, and test
    subsets = ["train", "valid", "test"]
    for subset in subsets:
        input_images_dir = os.path.join(base_dir, subset, "images")
        input_labels_dir = os.path.join(base_dir, subset, "labels")
        output_images_dir = os.path.join(base_dir + "_optimized", subset, "images")
        output_labels_dir = os.path.join(base_dir + "_optimized", subset, "labels")

        # Create output directories
        os.makedirs(output_images_dir, exist_ok=True)
        os.makedirs(output_labels_dir, exist_ok=True)

        # Process images
        if os.path.exists(input_images_dir):
            files = [f for f in os.listdir(input_images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            sampled_files = random.sample(files, int(len(files) * SAMPLE_FRACTION))

            for file in sampled_files:
                input_image_path = os.path.join(input_images_dir, file)
                output_image_path = os.path.join(output_images_dir, file)

                # Resize and compress image
                with Image.open(input_image_path) as img:
                    img_resized = img.resize(RESIZE_SIZE)
                    img_resized.save(output_image_path, "JPEG", quality=COMPRESSION_QUALITY)

                # Copy corresponding label file
                label_file = file.rsplit('.', 1)[0] + ".txt"
                input_label_path = os.path.join(input_labels_dir, label_file)
                output_label_path = os.path.join(output_labels_dir, label_file)
                if os.path.exists(input_label_path):
                    shutil.copy(input_label_path, output_label_path)

    # Create YAML file
    optimized_base_dir = base_dir + "_optimized"
    yaml_content = f"""
train: {optimized_base_dir}/train/images
val: {optimized_base_dir}/valid/images
test: {optimized_base_dir}/test/images
nc: {len(class_names)}  # Number of classes
names: {class_names}  # Class names
"""
    with open(yaml_file, 'w') as yaml_file_obj:
        yaml_file_obj.write(yaml_content)
    print(f"YAML file created: {yaml_file}")

# Process all datasets
for dataset_name, dataset_info in DATASETS.items():
    optimize_dataset(dataset_name, dataset_info)

print("All datasets optimized and YAML files created.")
