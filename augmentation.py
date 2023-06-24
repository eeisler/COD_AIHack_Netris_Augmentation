import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as T


def plot(imgs, with_orig=False, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [orig_img] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    plt.close()


def folder_save(file_name, imgs, transformation_type):
    jpg_folder_path = 'results/images'

    if not os.path.exists(jpg_folder_path):
        os.makedirs(jpg_folder_path)
        print(f"Directory '{jpg_folder_path}' created successfully.")
    else:
        print(f"Directory '{jpg_folder_path}' already exists.")

    if isinstance(imgs, list):
        for i, img in enumerate(imgs):
            img_path = f"{jpg_folder_path}/{file_name[:-4]}.{i + 1}.{transformation_type}.jpg"
            img.save(img_path)

        print(f"Saved {len(imgs)} images.\n")
    else:
        img_path = f"{jpg_folder_path}/{file_name[:-4]}.1.jpg"
        imgs.save(img_path)
        print("Saved 1 image.\n")


def gray(orig_img, file_name):
    transformation_type = 'gray'
    gray_img = T.Grayscale()(orig_img)
    plot([gray_img], cmap='gray')
    folder_save(file_name, gray_img, transformation_type)


def jitter(orig_img, file_name):
    transformation_type = 'filter'
    jitter = T.ColorJitter(brightness=.5, hue=.3)
    jitted_imgs = [jitter(orig_img) for _ in range(4)]
    plot(jitted_imgs)
    folder_save(file_name, jitted_imgs, transformation_type)


def blur(orig_img, file_name):
    transformation_type = 'blur'
    blurrer = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
    blurred_imgs = [blurrer(orig_img) for _ in range(4)]
    plot(blurred_imgs)
    folder_save(file_name, blurred_imgs, transformation_type)


def post(orig_img, file_name):
    transformation_type = 'post'
    posterizer = T.RandomPosterize(bits=2)
    posterized_imgs = [posterizer(orig_img) for _ in range(4)]
    plot(posterized_imgs)
    folder_save(file_name, posterized_imgs, transformation_type)


def sharp(orig_img, file_name):
    transformation_type = 'sharp'
    sharpness_adjuster = T.RandomAdjustSharpness(sharpness_factor=2)
    sharpened_imgs = [sharpness_adjuster(orig_img) for _ in range(4)]
    plot(sharpened_imgs)
    folder_save(file_name, sharpened_imgs, transformation_type)


def equalizer(orig_img, file_name):
    transformation_type = 'equalizer'
    equalizer = T.RandomEqualize()
    equalized_imgs = [equalizer(orig_img) for _ in range(4)]
    plot(equalized_imgs)
    folder_save(file_name, equalized_imgs, transformation_type)


def flipper(orig_img, file_name):
    transformation_type = 'flipper'
    hflipper = T.RandomHorizontalFlip(p=0.5)
    transformed_imgs = [hflipper(orig_img) for _ in range(4)]
    plot(transformed_imgs)
    folder_save(file_name, transformed_imgs, transformation_type)


def txt_change(txt_input_path):
    txt_result_path = r'C:\Users\codecamp\PycharmProjects\pythonProject1\results\labels'
    jpgs_path = r'C:\Users\codecamp\PycharmProjects\pythonProject1\results\images'

    print(os.listdir(jpgs_path))

    for jpg_file in os.listdir(jpgs_path):
        for txt_file in os.listdir(txt_input_path):
            old_txt_name = txt_file.split(".")[0]
            new_txt_name = os.path.splitext(jpg_file)[0]

            if old_txt_name == new_txt_name.split('.')[0]:
                print(f'{old_txt_name=}\t{new_txt_name=}')
                txt_file_path = os.path.join(txt_input_path, txt_file)
                with open(txt_file_path, 'r') as txt_file:
                    txt_data = txt_file.read()

                new_txt_path = os.path.join(txt_result_path, f"{new_txt_name}.txt")
                os.makedirs(os.path.dirname(new_txt_path), exist_ok=True)

                with open(new_txt_path, 'w') as new_txt_file:
                    new_txt_file.write(txt_data)


jpg_input_folder = r"C:\Users\codecamp\Downloads\train\images"
jpg_file_names = os.listdir(jpg_input_folder)
num_images = len(jpg_file_names)

txt_input_folder = r"C:\Users\codecamp\Downloads\train\labels"

for file_name in jpg_file_names:
    jpg_file_path = os.path.join(jpg_input_folder, file_name)
    if not os.path.isfile(jpg_file_path):
        continue
    print(file_name, end='\n')
    orig_img = Image.open(jpg_file_path)
    torch.manual_seed(0)

    gray(orig_img, file_name)
    jitter(orig_img, file_name)
    blur(orig_img, file_name)
    post(orig_img, file_name)
    sharp(orig_img, file_name)
    equalizer(orig_img, file_name)
    flipper(orig_img, file_name)

txt_change(txt_input_folder)
