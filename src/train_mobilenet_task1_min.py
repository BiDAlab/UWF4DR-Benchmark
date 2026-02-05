
import os
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC


# Optional import if splits are generated in memory
from prepare_split_task_1 import build_task1_datasets


# -------------------------------------------------
# Image preprocessing utilities
# -------------------------------------------------

def center_crop(image, crop_size=(800, 800)):
    h, w, _ = image.shape
    ch, cw = crop_size
    top = (h - ch) // 2
    left = (w - cw) // 2
    return image[top:top + ch, left:left + cw]


def color_normalization(image, blur_radius=9, amplification_factor=4, offset=128):
    r, g, b = image.split()

    r_blur = r.filter(ImageFilter.GaussianBlur(blur_radius))
    g_blur = g.filter(ImageFilter.GaussianBlur(blur_radius))
    b_blur = b.filter(ImageFilter.GaussianBlur(blur_radius))

    def normalize(c, c_blur):
        c = np.asarray(c, np.float32)
        c_blur = np.asarray(c_blur, np.float32)
        c = c - c_blur
        c = np.clip(c * amplification_factor + offset, 0, 255)
        return Image.fromarray(c.astype(np.uint8), mode="L")

    r = normalize(r, r_blur)
    g = normalize(g, g_blur)
    b = normalize(b, b_blur)

    return Image.merge("RGB", (r, g, b))


def load_and_preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.numpy_function(_preprocess_numpy, [img], tf.float32)
    img.set_shape((448, 448, 3))
    return img, label


def _preprocess_numpy(img):
    img = img.numpy()
    img = Image.fromarray(img)
    img = np.array(img)
    img = center_crop(img, (800, 800))
    img = tf.image.resize(img, (448, 448)).numpy()
    img = Image.fromarray(img.astype(np.uint8))
    img = color_normalization(img)
    img = np.asarray(img)
    img = preprocess_input(img)
    return img.astype(np.float32)


def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.rot90(image, tf.random.uniform([], 0, 4, tf.int32))
    return image, label


# -------------------------------------------------
# Model
# -------------------------------------------------

def build_model():
    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(448, 448, 3)
    )
    base.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = Dense(
        128,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    out = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base.input, outputs=out)

    model.compile(
        optimizer=Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", AUC(name="auc")]
    )

    return model


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Minimal training script for Task 1 using MobileNetV2."
    )

    parser.add_argument("--splits_dir", default=None)

    parser.add_argument("--task1_training_root", default=None)
    parser.add_argument("--task1_validation_root", default=None)
    parser.add_argument("--task23_training_root", default=None)
    parser.add_argument("--task23_validation_root", default=None)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)

    args = parser.parse_args()

    # Load datasets
    if args.splits_dir:
        datasets = {}
        for split in ["train", "validation"]:
            df = pd.read_csv(os.path.join(args.splits_dir, f"{split}.csv"))
            datasets[split] = list(zip(df.image_path, df.label))
    else:
        datasets = build_task1_datasets(
            args.task1_training_root,
            args.task1_validation_root,
            args.task23_training_root,
            args.task23_validation_root
        )

    # Build tf.data pipelines
    train_ds = tf.data.Dataset.from_tensor_slices(datasets["train"])
    train_ds = (
        train_ds
        .map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(256)
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = tf.data.Dataset.from_tensor_slices(datasets["validation"])
    val_ds = (
        val_ds
        .map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    model = build_model()
    model.summary()

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs
    )

    print("✅ Minimal training run completed.")


if __name__ == "__main__":
    main()
