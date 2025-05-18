import os
import shutil
import random

# Base paths
base_dir = "dataset"
images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")

# Output splits and ratios
splits = ["train", "val", "test"]
ratios = [0.7, 0.2, 0.1]

# Collect .png files with matching .json labels
image_files = [
    f for f in os.listdir(images_dir)
    if f.lower().endswith(".png") and os.path.exists(os.path.join(labels_dir, f.replace(".png", ".json")))
]

# Shuffle for randomness
random.shuffle(image_files)

# Compute split sizes
total = len(image_files)
train_end = int(ratios[0] * total)
val_end = train_end + int(ratios[1] * total)

splits_data = {
    "train": image_files[:train_end],
    "val": image_files[train_end:val_end],
    "test": image_files[val_end:]
}

# Create target folders and copy files
for split in splits:
    split_img_dir = os.path.join(base_dir, split, "images")
    split_lbl_dir = os.path.join(base_dir, split, "labels")
    os.makedirs(split_img_dir, exist_ok=True)
    os.makedirs(split_lbl_dir, exist_ok=True)

    for filename in splits_data[split]:
        # Full source paths
        img_src = os.path.join(images_dir, filename)
        lbl_src = os.path.join(labels_dir, filename.replace(".png", ".json"))

        # Full target paths
        img_dst = os.path.join(split_img_dir, filename)
        lbl_dst = os.path.join(split_lbl_dir, filename.replace(".png", ".json"))

        shutil.copy2(img_src, img_dst)
        shutil.copy2(lbl_src, lbl_dst)

print("✅ Dataset successfully split into train/val/test.")
