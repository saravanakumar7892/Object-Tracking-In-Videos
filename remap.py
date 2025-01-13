import os

# Define the class re-mapping dictionary
class_remap = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4,
    5: 5, 6: 6, 7: 7, 8: 8, 9: 9,
    10: 10, 11: 11, 12: 12, 13: 13, 14: 14,
    17: 15, 20: 16, 25: 17, 28: 18, 30: 19
}

# Function to remap a single label file
def remap_labels(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        class_id, *coords = line.split()
        class_id = int(class_id)
        if class_id in class_remap:
            new_class_id = class_remap[class_id]
            new_lines.append(f"{new_class_id} {' '.join(coords)}\n")
    with open(label_file, 'w') as f:
        f.writelines(new_lines)

# Function to process all label files in a folder
def process_folder(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):  # Assuming YOLO format annotations
                label_file = os.path.join(root, file)
                remap_labels(label_file)
                print(f"Processed {label_file}")

# Paths to your dataset folders
train_labels_path = "E:/Object_detection/datasets/custom_dataset/train/labels"
val_labels_path = "E:/Object_detection/datasets/custom_dataset/valid/labels"

# Process the training and validation folders
process_folder(train_labels_path)
process_folder(val_labels_path)

print("Re-mapping completed.")
