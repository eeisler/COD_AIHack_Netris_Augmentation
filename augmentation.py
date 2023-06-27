import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as T

jpg_input_path = input("Enter the path to input images folder: ")
txt_input_path = input("Enter the path to input labels folder: ")

downloads_folder = os.path.expanduser(r"~\Downloads")

txt_result_folder = r'results\labels'
txt_result_path = os.path.join(downloads_folder, txt_result_folder)
os.makedirs(txt_result_path, exist_ok=True)

jpg_result_folder = r'results\images'
jpg_result_path = os.path.join(downloads_folder, jpg_result_folder)
os.makedirs(jpg_result_path, exist_ok=True)


def folder_save(file_name, imgs, transformation_type):
    if isinstance(imgs, list):
        for i, img in enumerate(imgs):
            img_path = f"{jpg_result_path}/{file_name[:-4]}.{i + 1}.{transformation_type}.jpg"
            img.save(img_path)

        print(f"Saved {len(imgs)} images")
    else:
        img_path = f"{jpg_result_path}/{file_name[:-4]}.1.{transformation_type}.jpg"
        imgs.save(img_path)
        print("Saved 1 image")


def gray(orig_img, file_name):
    transformation_type = 'gray'
    gray_img = T.Grayscale()(orig_img)
    folder_save(file_name, gray_img, transformation_type)


def jitter(orig_img, file_name):
    transformation_type = 'filter'
    jitter = T.ColorJitter(brightness=.5, hue=.3)
    jitted_imgs = [jitter(orig_img) for _ in range(4)]
    folder_save(file_name, jitted_imgs, transformation_type)


def blur(orig_img, file_name):
    transformation_type = 'blur'
    blurrer = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
    blurred_imgs = [blurrer(orig_img) for _ in range(4)]
    folder_save(file_name, blurred_imgs, transformation_type)


def post(orig_img, file_name):
    transformation_type = 'post'
    posterizer = T.RandomPosterize(bits=2)
    posterized_imgs = [posterizer(orig_img) for _ in range(4)]
    folder_save(file_name, posterized_imgs, transformation_type)


def sharp(orig_img, file_name):
    transformation_type = 'sharp'
    sharpness_adjuster = T.RandomAdjustSharpness(sharpness_factor=2)
    sharpened_imgs = [sharpness_adjuster(orig_img) for _ in range(4)]
    folder_save(file_name, sharpened_imgs, transformation_type)


def equalizer(orig_img, file_name):
    transformation_type = 'equalizer'
    equalizer = T.RandomEqualize()
    equalized_imgs = [equalizer(orig_img) for _ in range(4)]
    folder_save(file_name, equalized_imgs, transformation_type)


def flipper(orig_img, file_name):
    transformation_type = 'flipper'
    hflipper = T.RandomHorizontalFlip(p=0.5)
    transformed_imgs = [hflipper(orig_img) for _ in range(4)]
    folder_save(file_name, transformed_imgs, transformation_type)


def txt_change(txt_input_path):
    jpg_result_file_names = os.listdir(jpg_result_path)
    txt_file_names = os.listdir(txt_input_path)

    for jpg_file in sorted(jpg_result_file_names):
        for txt_file in txt_file_names:
            old_txt_name = txt_file.split(".")[0]
            new_txt_name = os.path.splitext(jpg_file)[0]

            if old_txt_name == new_txt_name.split('.')[0]:
                txt_file_path = os.path.join(txt_input_path, txt_file)

                with open(txt_file_path, 'r') as txt_file:
                    txt_data = txt_file.read()

                new_txt_path = os.path.join(txt_result_path, f"{new_txt_name}.txt")
                os.makedirs(os.path.dirname(new_txt_path), exist_ok=True)

                with open(new_txt_path, 'w') as new_txt_file:
                    new_txt_file.write(txt_data)


jpg_input_file_names = os.listdir(jpg_input_path)

for file_name in jpg_input_file_names:
    jpg_file_path = os.path.join(jpg_input_path, file_name)
    if not os.path.isfile(jpg_file_path):
        continue
    print(f"\n{file_name} file in progress\n")
    orig_img = Image.open(jpg_file_path)
    torch.manual_seed(0)

    gray(orig_img, file_name)
    jitter(orig_img, file_name)
    blur(orig_img, file_name)
    post(orig_img, file_name)
    sharp(orig_img, file_name)
    equalizer(orig_img, file_name)
    flipper(orig_img, file_name)

txt_change(txt_input_path)

jpg_result_files = os.listdir(jpg_result_path)
jpg_file_count = len(jpg_result_files)

txt_result_files = os.listdir(txt_result_path)
txt_file_count = len(txt_result_files)

print(f"Images saved: {jpg_file_count}")
print(f"Labels saved: {txt_file_count}")

