import os
import yaml 


def check_dataset_classes(labels_path):
    max_class = -1
    label_files = [f for f in os.listdir(labels_path) if f.endswith('.txt')]

    for label_file in label_files:
        with open(os.path.join(labels_path, label_file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                class_id = int(line.split()[0])  # The class ID is the first value in YOLO format
                max_class = max(max_class, class_id)

    return max_class + 1  # Classes are 0-indexed, so add 1 for the total count

def check_yaml_classes(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data.get('nc', None)  # Get the 'nc' (number of classes) field

# Paths to your dataset
labels_path = "E:/Object_detection/datasets/coco2014/labels/train2014"  # Adjust to your dataset labels path
yaml_path = "E:/Object_detection/datasets/coco2014/coco2014.yaml"

# Check classes
detected_classes = check_dataset_classes(labels_path)
yaml_classes = check_yaml_classes(yaml_path)

print(f"Number of classes in the dataset: {detected_classes}")
print(f"Number of classes in the YAML file: {yaml_classes}")

if detected_classes != yaml_classes:
    print(f"Mismatch detected! Update the 'nc' value in {yaml_path} to {detected_classes}.")
