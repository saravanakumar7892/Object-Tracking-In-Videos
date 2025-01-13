import os

def count_class_instances(labels_dir):
    """
    Function to count instances of each class in label files.
    Arguments:
    - labels_dir: Path to the directory containing label files.
    
    Returns:
    - class_instance_counts: A dictionary with class IDs as keys and their respective counts as values.
    """
    class_instance_counts = {i: 0 for i in range(31)}  # Assuming 31 classes (0-30)
    
    # Loop through all label files and count class occurrences
    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):  # Only consider label files
            with open(os.path.join(labels_dir, label_file), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    class_id = int(line.split()[0])  # Class id is the first number in each line
                    class_instance_counts[class_id] += 1

    return class_instance_counts

def check_class_distribution_pascal():
    """
    Function to check class distribution for train and test datasets in PASCAL VOC style.
    """
    datasets = ['train', 'test']
    
    # Loop through each dataset (train, test)
    for dataset in datasets:
        print(f"\nChecking class distribution for {dataset} dataset:")

        if dataset == 'train':
            labels_dir = 'E:/Object_detection/datasets/PASCAL_VOC/train/labels'
        elif dataset == 'test':
            labels_dir = 'E:/Object_detection/datasets/custom_dataset/test/labels'

        if os.path.exists(labels_dir):
            class_instance_counts = count_class_instances(labels_dir)
            for class_id, count in class_instance_counts.items():
                print(f"Class {class_id}: {count} instances")
        else:
            print(f"Labels directory for {dataset} not found. Skipping...")

# Run the check for PASCAL VOC style dataset
check_class_distribution_pascal()
