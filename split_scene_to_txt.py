import os

def split_folder_names(directory, num_files):
    # Get the list of all folders in the specified directory
    folders = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    
    # Calculate the number of folders per file
    folders_per_file = len(folders) // num_files
    extra_folders = len(folders) % num_files
    
    start_index = 0
    
    # Write folder names to files
    for i in range(num_files):
        # Determine the number of folders for this file
        if i < extra_folders:
            count = folders_per_file + 1
        else:
            count = folders_per_file
        
        # Slice the folder list
        selected_folders = folders[start_index:start_index + count]
        start_index += count
        
        # Write to a file
        with open(f'/scratch/xi9/DATASET/DL3DV-960P-2K-split/folders_part_{i+1}.txt', 'w') as f:
            for folder in selected_folders:
                f.write(folder + '\n')
    
    print(f"Successfully split {len(folders)} folders into {num_files} files.")

# Example usage
directory_path = '/home/xi9/code/DATASET/DL3DV-960P-2K-Randominit'  # Replace with the path to your directory
num_output_files = 100  # Example: split into 4 files

split_folder_names(directory_path, num_output_files)
