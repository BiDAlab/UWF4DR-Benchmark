# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 17:37:40 2025

@author: pablo
"""

import os
import numpy as np
import cv2


def center_crop(image, crop_size=(800, 800)):
    
    height, width, _ = image.shape
    new_width, new_height = crop_size
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height
    return image[top:bottom, left:right]


def freq_transform_mag_clipped(img_np):
    img_np = np.array(img_np).astype(np.float32)
    freq_image_mag_clipped = np.zeros_like(img_np, dtype=np.float32)
    
    for i in range(3):
        f = np.fft.fft2(img_np[..., i])
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        clip_val = np.percentile(magnitude, 99)  
        mag_clipped = np.clip(magnitude, 0, clip_val)
        mag_clipped_norm = cv2.normalize(mag_clipped, None, 0, 255, cv2.NORM_MINMAX)
        freq_image_mag_clipped[..., i] = mag_clipped_norm

    return freq_image_mag_clipped.astype(np.uint8)


def process_images_from_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        img = cv2.imread(input_path)
        if img is None:
            print(f"No se pudo leer la imagen: {input_path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = center_crop(img, crop_size=(800, 800))
        img = cv2.resize(img,(224, 224))
        mag_img = freq_transform_mag_clipped(img)

        mag_filename = output_path
        cv2.imwrite(mag_filename, cv2.cvtColor(mag_img, cv2.COLOR_RGB2BGR))


input_dirs = ["C:/Users/pablo/Desktop/TFG_nuevo/Tarea 3/Data/train/class_0", "C:/Users/pablo/Desktop/TFG_nuevo/Tarea 3/Data/train/class_1", "C:/Users/pablo/Desktop/TFG_nuevo/Tarea 3/Data/val/class_0", "C:/Users/pablo/Desktop/TFG_nuevo/Tarea 3/Data/val/class_1", "C:/Users/pablo/Desktop/TFG_nuevo/Tarea 3/Data/test/class_0", "C:/Users/pablo/Desktop/TFG_nuevo/Tarea 3/Data/test/class_1"]
output_dirs = ["C:/Users/pablo/Desktop/TFG_nuevo/Tarea 3/Data Fourier/train/class_0", "C:/Users/pablo/Desktop/TFG_nuevo/Tarea 3/Data Fourier/train/class_1", "C:/Users/pablo/Desktop/TFG_nuevo/Tarea 3/Data Fourier/val/class_0", "C:/Users/pablo/Desktop/TFG_nuevo/Tarea 3/Data Fourier/val/class_1", "C:/Users/pablo/Desktop/TFG_nuevo/Tarea 3/Data Fourier/test/class_0", "C:/Users/pablo/Desktop/TFG_nuevo/Tarea 3/Data Fourier/test/class_1"]

for in_dir, out_dir in zip(input_dirs, output_dirs):
    process_images_from_folder(in_dir, out_dir)