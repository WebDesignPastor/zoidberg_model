import zipfile
import os

def zip_images(batch_path, zip_filename, num_images, keyword: str):
    if not os.path.exists(batch_path):
        print(f"Le dossier {batch_path} n'existe pas.")
        return

    image_files = [f for f in os.listdir(batch_path) if os.path.isfile(os.path.join(batch_path, f)) and keyword in f]

    num_images = min(num_images, len(image_files))

    with zipfile.ZipFile(zip_filename, 'w') as zip_file:
        for i in range(num_images):
            image_file = image_files[i]
            image_path = os.path.join(batch_path, image_file)
            zip_file.write(image_path, os.path.basename(image_path))

    print(f"Archive ZIP créée avec succès avec {num_images} images.")


batch_path = "/content/drive/MyDrive/ZoidBerg2.0 - T-Dev-810/chest_Xray/train/BACTERIA"
zip_filename = "/content/drive/MyDrive/ZoidBerg2.0 - T-Dev-810/chest_Xray/train/bacteria.zip"
num_images = 150
keyword = ""  # to select special filename
 
zip_images(batch_path, zip_filename, num_images, keyword)

