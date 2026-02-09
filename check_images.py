import os
from PIL import Image

# Path to your training dataset
train_dir = r'E:\canteen-system\sign_language_project\dataset\asl_alphabet_train'

for label in os.listdir(train_dir):
    folder = os.path.join(train_dir, label)
    count = 0
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        try:
            # Try opening the image
            Image.open(file_path).verify()  # verify() is safer
            count += 1
        except Exception as e:
            print("Broken:", file_path, e)
    print(f"{label}: {count} valid images")
