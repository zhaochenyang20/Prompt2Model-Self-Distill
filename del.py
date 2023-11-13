
import os

# Traverse the current directory and delete cache files and directories
for dirpath, dirnames, filenames in os.walk(".", topdown=True):
    # Delete optimizer
    for filename in filenames:
        if filename == "optimizer.pt":
            file_path = os.path.join(dirpath, filename)
            print(f"Deleting {file_path}")
            os.remove(file_path)

