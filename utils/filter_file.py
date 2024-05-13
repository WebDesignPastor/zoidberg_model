import os

base_dir = "/content/drive/MyDrive/ZoidBerg2.0 - T-Dev-810/chest_Xray"
pneumonia_dir = base_dir + "/train/PNEUMONIA/"
virus_dir = base_dir + "/train/VIRUS/"
bacteria_dir = base_dir + "/train/BACTERIA/"
os.mkdir(virus_dir)
os.mkdir(bacteria_dir)

dir_target = os.listdir(pneumonia_dir)

for dir in dir_target:
    if "virus" in dir:
        os.rename(pneumonia_dir + dir, virus_dir + dir)
    else:
        os.rename(pneumonia_dir + dir, bacteria_dir + dir)

print(base_dir)
print(os.listdir(virus_dir))
print(os.listdir(bacteria_dir))