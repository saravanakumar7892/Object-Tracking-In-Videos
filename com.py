import pandas as pd

# Function to load and check columns in the log data
def load_log(file_path):
    log_data = pd.read_csv(file_path)
    print(f"Columns in {file_path}: {log_data.columns.tolist()}")  # Print column names
    return log_data

# Function to clean and prepare the logs for comparison
def prepare_comparison(log1, log2):
    log1.columns = log1.columns.str.strip()  # Remove spaces from column names
    log2.columns = log2.columns.str.strip()  # Remove spaces from column names
    
    # If necessary, rename columns (e.g., 'Epoch' to 'epoch')
    log1.rename(columns={'Epoch': 'epoch'}, inplace=True)
    log2.rename(columns={'Epoch': 'epoch'}, inplace=True)
    
    # Merge logs on 'epoch'
    comparison_df = pd.merge(log1, log2, on='epoch', suffixes=('_exp28', '_exp26'))
    return comparison_df

# Function to display the comparison table
def display_comparison_table(comparison_df):
    print(comparison_df.to_string(index=False))

# Main function to compare the logs
def compare_training_logs(log_file1, log_file2):
    # Load the logs
    log1 = load_log(log_file1)
    log2 = load_log(log_file2)
    
    # Prepare the comparison data
    comparison_df = prepare_comparison(log1, log2)
    
    # Display the comparison table
    display_comparison_table(comparison_df)

# Example usage with the specified file paths
log_file1 = r'E:\Object_detection\yolov5\runs\train\exp28\results.csv'
log_file2 = r'E:\Object_detection\yolov5\runs\train\exp26\results.csv'
compare_training_logs(log_file1, log_file2)
