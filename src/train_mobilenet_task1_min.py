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

# Needed when generating splits in memory
from prepare_split_task_1 import build_task1_datasets


# =================================================
# Image preprocessing utilities
# =================================================

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


def _preprocess_numpy(img):
    img = Image.fromarray(img)
    img = np.array(img)

    img = center_crop(img, (800, 800))
    img = tf.image.resize(img, (448, 448)).numpy().astype(np.uint8)

    img = Image.fromarray(img)
    img = color_normalization(img)

    img = np.asarray(img)
    img = preprocess_input(img)

    return img.astype(np.float32)


def load_and_preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)

    img = tf.numpy_function(
        _preprocess_numpy,
        [img],
        Tout=tf.float32
    )
    img.set_shape((448, 448, 3))

    return img, tf.cast(label, tf.float32)


def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.rot90(image, tf.random.uniform([], 0, 4, tf.int32))
    return image, label


# =================================================
# Model
# =================================================

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


# =================================================
# Main
# =================================================

def main():
    parser = argparse.ArgumentParser(
        description="Minimal training script for Task 1 using MobileNetV2."
    )

    # Option A: use saved CSV splits
    parser.add_argument("--splits_dir", default=None)

    # Option B: generate splits in memory
    parser.add_argument("--task1_training_root", default=None)
    parser.add_argument("--task1_validation_root", default=None)
    parser.add_argument("--task23_training_root", default=None)
    parser.add_argument("--task23_validation_root", default=None)

    # Training params
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)

    # Optional model saving
    parser.add_argument(
        "--save_model",
        default=None,
        help="Optional path to save the trained model (.keras)"
    )

    args = parser.parse_args()

    # -------------------------------------------------
    # Load datasets
    # -------------------------------------------------

    if args.splits_dir is not None:
        print("📂 Loading dataset splits from CSV files")

        train_df = pd.read_csv(os.path.join(args.splits_dir, "train.csv"))
        val_df = pd.read_csv(os.path.join(args.splits_dir, "validation.csv"))

        train_paths = train_df["image_path"].tolist()
        train_labels = train_df["label"].tolist()

        val_paths = val_df["image_path"].tolist()
        val_labels = val_df["label"].tolist()

    else:
        print("🧠 Generating dataset splits in memory")

        datasets = build_task1_datasets(
            args.task1_training_root,
            args.task1_validation_root,
            args.task23_training_root,
            args.task23_validation_root,
        )

        train_paths, train_labels = zip(*datasets["train"])
        val_paths, val_labels = zip(*datasets["validation"])

        train_paths = list(train_paths)
        train_labels = list(train_labels)
        val_paths = list(val_paths)
        val_labels = list(val_labels)

    print(f"Train samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")

    # -------------------------------------------------
    # tf.data pipelines
    # -------------------------------------------------

    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_ds = (
        train_ds
        .map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(256)
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_ds = (
        val_ds
        .map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # -------------------------------------------------
    # Train
    # -------------------------------------------------

    model = build_model()
    model.summary()

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs
    )

    # -------------------------------------------------
    # Optional saving
    # -------------------------------------------------

    if args.save_model is not None:
        print(f"💾 Saving model to: {args.save_model}")
        model.save(args.save_model)

    print("✅ Minimal training run completed successfully.")


if __name__ == "__main__":
    main()
