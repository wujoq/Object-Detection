import albumentations as A
import cv2
import numpy as np
import os
import random
from glob import glob
import json

def read_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def augment_image(image):
    height, width = image.shape[:2]
    diagonal = int(np.sqrt(height**2 + width**2))
    padded_height, padded_width = diagonal, diagonal
    transform_pad = A.PadIfNeeded(min_height=padded_height, min_width=padded_width,
                                  border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    transform = A.Compose([
        transform_pad,
        A.Rotate(limit=(0, 360), border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0, 0), p=1.0),
    ])
    augmented = transform(image=image)
    return augmented['image']

def augment_background(image):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=1.0),
        A.RandomGamma(p=0.8),

        #A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT_101, value=(0, 0, 0), p=0.5),
        #A.RandomScale(scale_limit=0.1, p=0.5),
        #A.PadIfNeeded(min_height=image.shape[0], min_width=image.shape[1],
        #              border_mode=cv2.BORDER_REFLECT_101, value=(0, 0, 0)),
    ])
    augmented = transform(image=image)
    return augmented["image"]

def resize_to_fit(fg, bg):
    scale = random.uniform(0.2, 0.3)
    fg_h, fg_w = fg.shape[:2]
    bg_h, bg_w = bg.shape[:2]
    max_width = int(bg_w * scale)
    max_height = int(bg_h * scale)
    scale_factor = min(max_width / fg_w, max_height / fg_h, 1.0)
    new_size = (int(fg_w * scale_factor), int(fg_h * scale_factor))
    return cv2.resize(fg, new_size, interpolation=cv2.INTER_AREA)

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea)

def place_on_background(foreground, background, existing_bboxes, class_id, label_list):
    bg_h, bg_w = background.shape[:2]
    fg_h, fg_w = foreground.shape[:2]

    for _ in range(100):
        x_offset = random.randint(0, bg_w - fg_w)
        y_offset = random.randint(0, bg_h - fg_h)
        new_box = [x_offset, y_offset, fg_w, fg_h]

        if all(calculate_iou(new_box, b) < 0.01 for b in existing_bboxes):
            fg_rgb = foreground[:, :, :3]
            fg_alpha = foreground[:, :, 3] / 255.0
            roi = background[y_offset:y_offset + fg_h, x_offset:x_offset + fg_w]
            blended = (fg_rgb * fg_alpha[:, :, None] + roi * (1 - fg_alpha[:, :, None])).astype(np.uint8)
            background[y_offset:y_offset + fg_h, x_offset:x_offset + fg_w] = blended

            existing_bboxes.append(new_box)
            label_list.append({
                "class_id": class_id,
                "bbox": new_box
            })
            return background

    raise Exception("Failed to place object without overlap.")

def save_image(image, path):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image)

def draw_boxes(image, boxes):
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return image

# Collect files
backgrounds = glob("backgrounds/*.jpg")
drivers = glob("Data_for_augmentation/driver*.png")
kombs = glob("Data_for_augmentation/komb*.png")
swords = glob("Data_for_augmentation/sword*.png")

# Output folders
label_dir = "dataset/labels"
image_dir = "dataset/images"
image_boxes_dir = "dataset/images_boxes"

os.makedirs(label_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)
os.makedirs(image_boxes_dir, exist_ok=True)

for bg_index, bg_path in enumerate(backgrounds):
    bg_base = os.path.splitext(os.path.basename(bg_path))[0]
    #bg_image = augment_background(read_image(bg_path))

    for i in range(1, 21):
        #composite = bg_image.copy()
        bboxes = []
        labels = []
        composite = augment_background(read_image(bg_path))
        objects = [
            (random.choice(drivers), 0),
            (random.choice(kombs), 1),
            (random.choice(swords), 2)
        ]

        for obj_path, class_id in objects:
            obj = read_image(obj_path)
            obj_aug = resize_to_fit(augment_image(obj), composite)
            composite = place_on_background(obj_aug, composite, bboxes, class_id, labels)

        filename = f"composite_{bg_base}_{i:02}"
        composite_with_boxes = draw_boxes(composite.copy(), [obj["bbox"] for obj in labels])
        save_image(composite_with_boxes, os.path.join(image_boxes_dir, f"{filename}_debug.png"))

        save_image(composite, os.path.join(image_dir, f"{filename}.png"))
        with open(os.path.join(label_dir, f"{filename}.json"), "w") as f:
            json.dump({"objects": labels}, f, indent=2)

print("✅ Wszystkie kompozycje i etykiety JSON zapisane w 'dataset/'")
