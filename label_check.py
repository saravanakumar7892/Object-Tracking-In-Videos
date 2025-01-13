import os

def validate_yolo_labels(label_dir, output_invalid_file, valid_classes):
    """
    Validate YOLO label files and check for invalid labels.

    Args:
    - label_dir (str): Path to the folder containing label files.
    - output_invalid_file (str): Path to save invalid labels information.
    - valid_classes (range): Range of valid class IDs.

    Returns:
    - instance_counts (dict): A dictionary with class IDs and their instance counts.
    """
    instance_counts = {class_id: 0 for class_id in valid_classes}
    invalid_labels = []

    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            label_path = os.path.join(label_dir, label_file)
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, start=1):
                try:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        raise ValueError("Incorrect number of values (expected 5).")
                    
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])
                    
                    # Validate class ID
                    if class_id not in valid_classes:
                        raise ValueError(f"Invalid class ID: {class_id}")
                    
                    # Validate bounding box values
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                        raise ValueError(f"Bounding box values out of range: {x_center}, {y_center}, {width}, {height}")

                    # Increment instance count for valid labels
                    instance_counts[class_id] += 1
                
                except Exception as e:
                    invalid_labels.append(f"{label_file} (Line {line_num}): {e}")
    
    # Save invalid labels to a file
    if invalid_labels:
        with open(output_invalid_file, 'w') as f:
            f.write("\n".join(invalid_labels))
        print(f"Invalid labels saved to {output_invalid_file}")
    else:
        print("All labels are valid.")

    return instance_counts

# === Example Usage ===
label_dir = "E:/Object_detection/datasets/custom_dataset/train/labels"  # Update with your labels folder path
output_invalid_file = "invalid_labels.txt"
valid_classes = range(31)  # Adjust based on your `nc` value (0 to 30 for nc=31)

# Validate labels and get instance counts
instance_counts = validate_yolo_labels(label_dir, output_invalid_file, valid_classes)

# Print instance counts per class
print("Class-wise instance counts:")
for class_id, count in instance_counts.items():
    print(f"Class {class_id}: {count} instances")
