from collections import Counter
import os

label_path = "E:/Object_detection/datasets/custom_dataset/train/labels"
class_counts = Counter()

# Count instances per class
for label_file in os.listdir(label_path):
    if label_file.endswith(".txt"):
        with open(os.path.join(label_path, label_file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                class_id = int(line.split()[0])
                class_counts[class_id] += 1

# Print class distribution
print("Class Distribution:")
for class_id, count in class_counts.items():
    print(f"Class {class_id}: {count} instances")
