import os

# Define a function to check if a directory exists, and if not, create it.
def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        return f"Directory created: {directory_path}"
    else:
        return f"Directory already exists: {directory_path}"

# For demonstration, let's use a sample directory path.
sample_directory_path = "/mnt/data/picture_files"

# Check and create the directory if it doesn't exist.
directory_check_result = ensure_directory_exists(sample_directory_path)
directory_check_result
