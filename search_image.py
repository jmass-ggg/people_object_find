import os
import re
import pickle
import faiss
import torch
import clip
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from pathlib import Path
from ultralytics import YOLO

from plate_ocr import PlateOCR


INDEX_FILE = "image_index.faiss"
METADATA_FILE = "image_metadata.pkl"

QUERY_DIR = Path("query")
RESULT_DIR = Path("result")

TOP_K = 5

# Search more results first, then rerank using plate number
RERANK_LIMIT = 50

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".JPG", ".JPEG", ".PNG")

CLIP_MODEL_NAME = "ViT-B/32"

YOLO_MODEL_NAME = "yolov8n.pt"
YOLO_CONFIDENCE = 0.25

PREFER_NON_PERSON = True

ENABLE_PLATE_OCR = True

VEHICLE_CLASSES = {
    "car",
    "motorcycle",
    "bus",
    "truck"
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

print("Loading CLIP model...")
clip_model, preprocess = clip.load(CLIP_MODEL_NAME, device=device)
clip_model.eval()

print("Loading YOLO model...")
yolo_model = YOLO(YOLO_MODEL_NAME)

plate_reader = PlateOCR(enabled=ENABLE_PLATE_OCR, gpu=torch.cuda.is_available())


def open_image_correctly(image_path):
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.exif_transpose(image)
    return image


def safe_filename(filename):
    filename = re.sub(r"[^A-Za-z0-9._-]+", "_", filename)
    return filename[:120]


def get_class_name(names, class_id):
    if isinstance(names, dict):
        return names.get(class_id, str(class_id))
    return names[class_id]


def find_query_image():
    preferred = QUERY_DIR / "query.jpg"

    if preferred.exists():
        return preferred

    if not QUERY_DIR.exists():
        raise FileNotFoundError("query folder not found. Create it using: mkdir query")

    images = []

    for file in QUERY_DIR.iterdir():
        if file.is_file() and file.name.lower().endswith(VALID_EXTENSIONS):
            images.append(file)

    if len(images) == 0:
        raise FileNotFoundError("No image found inside query folder.")

    return images[0]


def detect_and_crop_primary_object(image_path):
    original_image = open_image_correctly(image_path)
    img_w, img_h = original_image.size

    try:
        results = yolo_model.predict(
            source=str(image_path),
            conf=YOLO_CONFIDENCE,
            verbose=False
        )

        if len(results) == 0:
            return original_image, {
                "detected_object": "original",
                "yolo_confidence": 0.0,
                "used_original": True,
                "bbox": None
            }

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return original_image, {
                "detected_object": "original",
                "yolo_confidence": 0.0,
                "used_original": True,
                "bbox": None
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
                "score": area * confidence
            })

        non_person_candidates = [
            c for c in candidates if c["class_name"] != "person"
        ]

        if PREFER_NON_PERSON and len(non_person_candidates) > 0:
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
            "bbox": [x1, y1, x2, y2]
        }

        return cropped_image, info

    except Exception as e:
        print("YOLO failed for:", image_path)
        print("Reason:", e)

        return original_image, {
            "detected_object": "original",
            "yolo_confidence": 0.0,
            "used_original": True,
            "bbox": None
        }


def get_clip_embedding_from_pil(image):
    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = clip_model.encode_image(image_input)

    features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy().astype("float32")


def score_to_percentage(score):
    percentage = float(score) * 100

    if percentage < 0:
        percentage = 0

    if percentage > 100:
        percentage = 100

    return round(percentage, 2)


def calculate_final_score(visual_score, query_plate, candidate_plate):
    """
    Visual score comes from CLIP.
    Plate score boosts vehicle matches when plate number matches.
    """
    if query_plate and candidate_plate:
        if query_plate == candidate_plate:
            plate_score = 1.0
        else:
            plate_score = 0.0

        final_score = (0.65 * visual_score) + (0.35 * plate_score)
        return final_score

    return visual_score


def save_labeled_image(image, output_path, lines):
    image = image.convert("RGB")

    width, height = image.size
    banner_height = 30 + (len(lines) * 22)

    new_image = Image.new("RGB", (width, height + banner_height), "white")
    new_image.paste(image, (0, banner_height))

    draw = ImageDraw.Draw(new_image)

    y = 10

    for line in lines:
        draw.text((10, y), line, fill="black")
        y += 22

    new_image.save(output_path)


def main():
    RESULT_DIR.mkdir(exist_ok=True)

    if not os.path.exists(INDEX_FILE):
        raise FileNotFoundError("image_index.faiss not found. Run python build_index.py first.")

    if not os.path.exists(METADATA_FILE):
        raise FileNotFoundError("image_metadata.pkl not found. Run python build_index.py first.")

    index = faiss.read_index(INDEX_FILE)

    with open(METADATA_FILE, "rb") as f:
        metadata_records = pickle.load(f)

    query_image_path = find_query_image()

    print("Query image:", query_image_path)

    query_crop, query_info = detect_and_crop_primary_object(query_image_path)

    query_object = query_info["detected_object"]
    query_yolo_conf = query_info["yolo_confidence"]

    query_plate = None
    query_masked_plate = None
    query_plate_conf = 0.0

    if query_object in VEHICLE_CLASSES:
        query_plate, query_masked_plate, query_plate_conf = plate_reader.extract_from_pil(query_crop)

    query_embedding = get_clip_embedding_from_pil(query_crop)

    query_crop_path = RESULT_DIR / "query_cropped.jpg"

    query_lines = [
        f"Query detected: {query_object}",
        f"YOLO confidence: {round(query_yolo_conf * 100, 2)}%"
    ]

    if query_masked_plate:
        query_lines.append(f"Plate: {query_masked_plate}")

    save_labeled_image(query_crop, query_crop_path, query_lines)

    search_k = min(RERANK_LIMIT, index.ntotal)

    scores, indices = index.search(query_embedding, search_k)

    reranked_results = []

    for raw_rank, idx in enumerate(indices[0]):
        if idx == -1:
            continue

        visual_score = float(scores[0][raw_rank])
        metadata = metadata_records[idx]

        candidate_plate = metadata.get("plate_number")

        final_score = calculate_final_score(
            visual_score=visual_score,
            query_plate=query_plate,
            candidate_plate=candidate_plate
        )

        reranked_results.append({
            "metadata": metadata,
            "visual_score": visual_score,
            "final_score": final_score
        })

    reranked_results = sorted(
        reranked_results,
        key=lambda x: x["final_score"],
        reverse=True
    )

    top_results = reranked_results[:TOP_K]

    result_lines = []
    html_items = []

    result_lines.append(f"Query image: {query_image_path}")
    result_lines.append(f"Query detected object: {query_object}")
    result_lines.append(f"Query YOLO confidence: {round(query_yolo_conf * 100, 2)}%")
    result_lines.append(f"Query masked plate: {query_masked_plate}")
    result_lines.append("")
    result_lines.append("Top results:")
    result_lines.append("")

    print("\nTop Matching Results:\n")

    for rank, item in enumerate(top_results, start=1):
        metadata = item["metadata"]

        image_path = metadata["image_path"]
        detected_object = metadata.get("detected_object")
        masked_plate = metadata.get("masked_plate")

        visual_percentage = score_to_percentage(item["visual_score"])
        final_percentage = score_to_percentage(item["final_score"])

        matched_crop, matched_info = detect_and_crop_primary_object(image_path)

        original_name = Path(image_path).name
        output_filename = safe_filename(
            f"match_{rank}_{final_percentage}_percent_{original_name}"
        )

        output_path = RESULT_DIR / output_filename

        label_lines = [
            f"Rank #{rank}",
            f"Final match: {final_percentage}%",
            f"Visual match: {visual_percentage}%",
            f"Detected: {detected_object}"
        ]

        if masked_plate:
            label_lines.append(f"Plate: {masked_plate}")

        save_labeled_image(matched_crop, output_path, label_lines)

        print(f"Rank {rank}")
        print("Image:", image_path)
        print("Detected:", detected_object)
        print("Visual Similarity:", visual_percentage, "%")
        print("Final Match:", final_percentage, "%")

        if masked_plate:
            print("Plate:", masked_plate)

        print("Saved:", output_path)
        print()

        result_lines.append(f"Rank {rank}")
        result_lines.append(f"Image: {image_path}")
        result_lines.append(f"Detected: {detected_object}")
        result_lines.append(f"Visual Similarity: {visual_percentage}%")
        result_lines.append(f"Final Match: {final_percentage}%")
        result_lines.append(f"Plate: {masked_plate}")
        result_lines.append(f"Saved: {output_path}")
        result_lines.append("")

        html_items.append(f"""
        <div style="border:1px solid #ccc; padding:15px; margin-bottom:25px;">
            <h2>Rank {rank} - Final Match: {final_percentage}%</h2>
            <p><b>Visual Similarity:</b> {visual_percentage}%</p>
            <p><b>Detected:</b> {detected_object}</p>
            <p><b>Plate:</b> {masked_plate}</p>
            <p>{image_path}</p>
            <img src="{output_filename}" width="400">
        </div>
        """)

    txt_path = RESULT_DIR / "results.txt"

    with open(txt_path, "w") as f:
        f.write("\n".join(result_lines))

    html_path = RESULT_DIR / "results.html"

    html_content = f"""
    <html>
    <head>
        <title>Lost and Found AI Results</title>
    </head>
    <body style="font-family:Arial; padding:20px;">
        <h1>AI-Powered Lost and Found Matching Results</h1>

        <h2>Query Image</h2>
        <p><b>Image:</b> {query_image_path}</p>
        <p><b>Detected Object:</b> {query_object}</p>
        <p><b>YOLO Confidence:</b> {round(query_yolo_conf * 100, 2)}%</p>
        <p><b>Plate:</b> {query_masked_plate}</p>
        <img src="query_cropped.jpg" width="400">

        <hr>

        <h2>Top Matches</h2>
        {''.join(html_items)}
    </body>
    </html>
    """

    with open(html_path, "w") as f:
        f.write(html_content)

    print("Results saved successfully.")
    print("Open this file:")
    print(html_path)


if __name__ == "__main__":
    main()