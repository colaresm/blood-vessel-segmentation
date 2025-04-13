import os
masks_path_train = "data/train/mask"
images_path_train = "data/train/image"

masks_path_test = "data/test/mask"
images_path_test = "data/test/image"



def list_files(folder):
    try:
        return [file for file in os.listdir(folder) if os.path.isfile(os.path.join(folder, file))]
    except FileNotFoundError:
        print("Error: Folder not found.")
        return []
    except PermissionError:
        print("Error: Permission denied to access the folder.")
        return []