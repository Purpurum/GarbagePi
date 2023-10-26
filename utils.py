import os

def get_files_in_directory(directory_path):
    filenames = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    filepaths = [os.path.join(directory_path, f) for f in filenames]
    return filepaths, filenames