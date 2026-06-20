import re
from pathlib import Path

import clip
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from ultralytics import YOLO


VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}

VEHICLE_CLASSES = {
    "car",
    "motorcycle",
    "bus",
    "truck",
}


def open_image_correctly(image_path):
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def is_image_file(path):
    path = Path(path)
    return path.is_file() and path.suffix.lower() in VALID_EXTENSIONS


def get_all_image_paths(dataset_dir):
    dataset_dir = Path(dataset_dir)

    if not dataset_dir.exists():
        return []

    return sorted([
        path for path in dataset_dir.rglob("*")
        if is_image_file(path)
    ])


def safe_filename(filename):
    filename = re.sub(r"[^A-Za-z0-9._-]+", "_", filename)
    return filename[:120]


def get_class_name(names, class_id):
    if isinstance(names, dict):
        return names.get(class_id, str(class_id))

    return names[class_id]


def score_to_percentage(score):
    percentage = float(score) * 100.0
    percentage = max(0.0, min(100.0, percentage))
    return round(percentage, 2)


def crop_from_metadata(image_path, metadata):
    image = open_image_correctly(image_path)
    bbox = metadata.get("bbox")

    if not bbox:
        return image

    try:
        x1, y1, x2, y2 = [int(v) for v in bbox]

        width, height = image.size

        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))

        if x2 <= x1 or y2 <= y1:
            return image

        return image.crop((x1, y1, x2, y2))

    except Exception:
        return image


def save_labeled_image(image, output_path, lines):
    image = image.convert("RGB")

    width, height = image.size
    banner_height = 35 + len(lines) * 24

    new_image = Image.new("RGB", (width, height + banner_height), "white")
    new_image.paste(image, (0, banner_height))

    draw = ImageDraw.Draw(new_image)

    y = 10
    for line in lines:
        draw.text((10, y), str(line), fill="black")
        y += 24

    new_image.save(output_path)


def get_color_histogram_from_pil(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_np = np.array(image)
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

    hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        None,
        [16, 8, 8],
        [0, 180, 0, 256, 0, 256],
    )

    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype("float32")


def histogram_similarity(hist_a, hist_b):
    if hist_a is None or hist_b is None:
        return 0.0

    hist_a = np.array(hist_a, dtype="float32")
    hist_b = np.array(hist_b, dtype="float32")

    if hist_a.size == 0 or hist_b.size == 0:
        return 0.0

    score = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)

    score = (score + 1.0) / 2.0
    score = max(0.0, min(1.0, float(score)))

    return score


class VisionCore:
    def __init__(
        self,
        clip_model_name="ViT-B/32",
        yolo_model_name="yolov8n.pt",
        yolo_confidence=0.25,
        prefer_non_person=True,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.yolo_confidence = yolo_confidence
        self.prefer_non_person = prefer_non_person

        print("Using device:", self.device)

        print("Loading CLIP model...")
        self.clip_model, self.preprocess = clip.load(
            clip_model_name,
            device=self.device,
        )
        self.clip_model.eval()

        print("Loading YOLO model...")
        self.yolo_model = YOLO(yolo_model_name)

    def detect_and_crop_primary_object(self, image_path):
        original_image = open_image_correctly(image_path)
        img_w, img_h = original_image.size

        try:
            results = self.yolo_model.predict(
                source=np.array(original_image),
                conf=self.yolo_confidence,
                verbose=False,
            )

            if len(results) == 0:
                return original_image, {
                    "detected_object": "original",
                    "yolo_confidence": 0.0,
                    "used_original": True,
                    "bbox": None,
                }

            result = results[0]

            if result.boxes is None or len(result.boxes) == 0:
                return original_image, {
                    "detected_object": "original",
                    "yolo_confidence": 0.0,
                    "used_original": True,
                    "bbox": None,
                }

            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            names = result.names

            candidates = []

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box

                confidence = float(confs[i])
                class_id = int(classes[i])
                class_name = get_class_name(names, class_id)

                area = max(0, x2 - x1) * max(0, y2 - y1)

                candidates.append({
                    "box": box,
                    "confidence": confidence,
                    "class_name": class_name,
                    "score": area * confidence,
                })

            non_person_candidates = [
                c for c in candidates
                if c["class_name"] != "person"
            ]

            if self.prefer_non_person and len(non_person_candidates) > 0:
                selected = max(non_person_candidates, key=lambda x: x["score"])
            else:
                selected = max(candidates, key=lambda x: x["score"])

            x1, y1, x2, y2 = selected["box"]

            box_w = x2 - x1
            box_h = y2 - y1

            pad_x = box_w * 0.10
            pad_y = box_h * 0.10

            x1 = int(max(0, x1 - pad_x))
            y1 = int(max(0, y1 - pad_y))
            x2 = int(min(img_w, x2 + pad_x))
            y2 = int(min(img_h, y2 + pad_y))

            cropped_image = original_image.crop((x1, y1, x2, y2))

            info = {
                "detected_object": selected["class_name"],
                "yolo_confidence": selected["confidence"],
                "used_original": False,
                "bbox": [x1, y1, x2, y2],
            }

            return cropped_image, info

        except Exception as e:
            print("YOLO failed for:", image_path)
            print("Reason:", e)

            return original_image, {
                "detected_object": "original",
                "yolo_confidence": 0.0,
                "used_original": True,
                "bbox": None,
            }

    def get_clip_embedding_from_pil(self, image):
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.clip_model.encode_image(image_input)

        features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy().astype("float32")