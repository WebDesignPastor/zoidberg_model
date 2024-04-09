from google.colab import drive

drive.mount('/content/drive')
base_dir = "/content/drive/MyDrive/ZoidBerg2.0 - T-Dev-810/chest_Xray"

from PIL import Image
import os

sets = ["test", "val", "train"]
types = ["PNEUMONIA", "NORMAL"]
images = {}
for set_name in sets:
    images[f"{set_name}_virus"] = []
    images[f"{set_name}_bacteria"] = []
    images[f"{set_name}_normal"] = []
print("Loading images...")
for set_name in sets:
    for type_name in types:
        path = os.path.join(base_dir, set_name, type_name)
        for filename in os.listdir(path):
            if filename.endswith(".jpeg"):
                image = Image.open(os.path.join(path, filename))
                if "virus" in filename:
                    images[f"{set_name}_virus"].append(image)
                elif "bacteria" in filename:
                    images[f"{set_name}_bacteria"].append(image)
                else:
                    images[f"{set_name}_normal"].append(image)
for key in images:
    print(f"{key}: {len(images[key])} images loaded")