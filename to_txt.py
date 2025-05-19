import os
import json

# Input and output directories
splits = ['train', 'val', 'test']
for split in splits:
    json_dir = f'dataset/{split}/labels'
    txt_dir = f'dataset/{split}/labels_txt'
    os.makedirs(txt_dir, exist_ok=True)

    for filename in os.listdir(json_dir):
        if not filename.endswith('.json'):
            continue

        with open(os.path.join(json_dir, filename), 'r') as f:
            data = json.load(f)

        txt_filename = os.path.splitext(filename)[0] + '.txt'
        txt_path = os.path.join(txt_dir, txt_filename)

        with open(txt_path, 'w') as out:
            for obj in data['objects']:
                class_id = obj['class_id']
                x, y, w, h = obj['bbox']

                # Convert to YOLO format (normalized center x/y, width, height)
                img_path = f"dataset/{split}/images/{os.path.splitext(filename)[0]}.png"
                if not os.path.exists(img_path):
                    continue  # Skip if image doesn't exist
                import cv2
                img = cv2.imread(img_path)
                height, width = img.shape[:2]

                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                w_norm = w / width
                h_norm = h / height

                out.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

print("✅ All JSON annotations converted to YOLO TXT format.")
