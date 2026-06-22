import os

def list_all_files_and_folders(path):
    for root, dirs, files in os.walk(path):
        print(f"Folder: {root}")
        
        for d in dirs:
            print(f"  [DIR]  {d}")
        
        for f in files:
            print(f"  [FILE] {f}")

# change this to your target directory
list_all_files_and_folders(".")