import argparse
import html
import os
import pickle
from pathlib import Path

import faiss

from sensitive_ocr import SensitiveOCR
from vision_core import (
    VEHICLE_CLASSES,
    VisionCore,
    crop_from_metadata,
    get_color_histogram_from_pil,
    histogram_similarity,
    is_image_file,
    safe_filename,
    save_labeled_image,
    score_to_percentage,
)


DEFAULT_INDEX_FILE = "image_index.faiss"
DEFAULT_METADATA_FILE = "image_metadata.pkl"

DEFAULT_QUERY_DIR = "query"
DEFAULT_RESULT_DIR = "result"


def find_query_image(query_dir, query_image=None):
    query_dir = Path(query_dir)

    if query_image:
        query_image = Path(query_image)

        if not query_image.exists():
            raise FileNotFoundError(f"Query image not found: {query_image}")

        return query_image

    preferred = query_dir / "query.jpg"

    if preferred.exists():
        return preferred

    if not query_dir.exists():
        raise FileNotFoundError(
            "query folder not found. Create it using: mkdir query"
        )

    images = [
        file for file in query_dir.iterdir()
        if is_image_file(file)
    ]

    if len(images) == 0:
        raise FileNotFoundError("No image found inside query folder.")

    return sorted(images)[0]


def load_index_and_metadata(index_file, metadata_file):
    if not os.path.exists(index_file):
        raise FileNotFoundError(
            f"{index_file} not found. Run python build_index.py first."
        )

    if not os.path.exists(metadata_file):
        raise FileNotFoundError(
            f"{metadata_file} not found. Run python build_index.py first."
        )

    index = faiss.read_index(index_file)

    with open(metadata_file, "rb") as f:
        metadata_records = pickle.load(f)

    if index.ntotal != len(metadata_records):
        raise RuntimeError(
            "Index and metadata count mismatch. Rebuild the index."
        )

    return index, metadata_records


def save_metadata(metadata_file, metadata_records):
    with open(metadata_file, "wb") as f:
        pickle.dump(metadata_records, f)


def calculate_final_score(
    visual_score,
    color_score,
    query_number_hash,
    candidate_number_hash,
):
    visual_score = max(0.0, min(1.0, float(visual_score)))
    color_score = max(0.0, min(1.0, float(color_score)))

    if query_number_hash and candidate_number_hash:
        number_score = 1.0 if query_number_hash == candidate_number_hash else 0.0

        return (
            0.50 * visual_score
            + 0.20 * color_score
            + 0.30 * number_score
        )

    return (
        0.75 * visual_score
        + 0.25 * color_score
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search matching lost-and-found images."
    )

    parser.add_argument("--index-file", default=DEFAULT_INDEX_FILE)
    parser.add_argument("--metadata-file", default=DEFAULT_METADATA_FILE)

    parser.add_argument("--query-dir", default=DEFAULT_QUERY_DIR)
    parser.add_argument("--query-image", default=None)

    parser.add_argument("--result-dir", default=DEFAULT_RESULT_DIR)

    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--yolo-confidence", type=float, default=0.25)

    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rerank-limit", type=int, default=50)

    parser.add_argument(
        "--enable-ocr",
        action="store_true",
        help="Enable PaddleOCR for plates/passports/national IDs.",
    )

    parser.add_argument(
        "--ocr-candidates",
        type=int,
        default=0,
        help="How many top candidates to run OCR on. Use small number on CPU.",
    )

    parser.add_argument(
        "--allow-different-class",
        action="store_true",
        help="Allow matches with different YOLO class.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    result_dir = Path(args.result_dir)
    result_dir.mkdir(exist_ok=True)

    index, metadata_records = load_index_and_metadata(
        args.index_file,
        args.metadata_file,
    )

    query_image_path = find_query_image(
        args.query_dir,
        args.query_image,
    )

    print("Query image:", query_image_path)

    vision = VisionCore(
        clip_model_name=args.clip_model,
        yolo_model_name=args.yolo_model,
        yolo_confidence=args.yolo_confidence,
        prefer_non_person=True,
    )

    ocr_reader = SensitiveOCR(
        enabled=args.enable_ocr,
    )

    query_crop, query_info = vision.detect_and_crop_primary_object(query_image_path)

    query_object = query_info["detected_object"]
    query_yolo_conf = query_info["yolo_confidence"]

    query_embedding = vision.get_clip_embedding_from_pil(query_crop)
    query_color_histogram = get_color_histogram_from_pil(query_crop)

    query_number_hash = None
    query_masked_number = None
    query_number_type = None

    if args.enable_ocr:
        query_purpose = "plate" if query_object in VEHICLE_CLASSES else "document"

        query_ocr = ocr_reader.extract_from_pil(
            query_crop,
            purpose=query_purpose,
        )

        query_number_hash = query_ocr["number_hash"]
        query_masked_number = query_ocr["masked_number"]
        query_number_type = query_ocr["number_type"]

    query_crop_path = result_dir / "query_cropped.jpg"

    query_lines = [
        f"Query detected: {query_object}",
        f"YOLO confidence: {round(query_yolo_conf * 100, 2)}%",
    ]

    if query_masked_number:
        query_lines.append(f"Number: {query_masked_number}")
        query_lines.append(f"Number type: {query_number_type}")

    save_labeled_image(query_crop, query_crop_path, query_lines)

    search_k = min(args.rerank_limit, index.ntotal)

    scores, indices = index.search(query_embedding, search_k)

    reranked_results = []
    metadata_changed = False

    for raw_rank, idx in enumerate(indices[0]):
        if idx == -1:
            continue

        metadata = metadata_records[idx]

        candidate_object = metadata.get("detected_object")

        if not args.allow_different_class and query_object != "original":
            if candidate_object != query_object:
                continue

        visual_score = float(scores[0][raw_rank])

        candidate_color_histogram = metadata.get("color_histogram")
        color_score = histogram_similarity(
            query_color_histogram,
            candidate_color_histogram,
        )

        candidate_number_hash = metadata.get("number_hash")
        candidate_masked_number = metadata.get("masked_number")
        candidate_number_type = metadata.get("number_type")

        if (
            args.enable_ocr
            and query_number_hash
            and not candidate_number_hash
            and raw_rank < args.ocr_candidates
        ):
            candidate_crop = crop_from_metadata(
                metadata["image_path"],
                metadata,
            )

            candidate_purpose = "plate" if candidate_object in VEHICLE_CLASSES else "document"

            candidate_ocr = ocr_reader.extract_from_pil(
                candidate_crop,
                purpose=candidate_purpose,
            )

            candidate_number_hash = candidate_ocr["number_hash"]
            candidate_masked_number = candidate_ocr["masked_number"]
            candidate_number_type = candidate_ocr["number_type"]

            metadata["number_hash"] = candidate_number_hash
            metadata["masked_number"] = candidate_masked_number
            metadata["number_type"] = candidate_number_type
            metadata["number_confidence"] = candidate_ocr["confidence"]

            metadata_changed = True

        final_score = calculate_final_score(
            visual_score=visual_score,
            color_score=color_score,
            query_number_hash=query_number_hash,
            candidate_number_hash=candidate_number_hash,
        )

        reranked_results.append({
            "index": idx,
            "metadata": metadata,
            "visual_score": visual_score,
            "color_score": color_score,
            "final_score": final_score,
        })

    if metadata_changed:
        save_metadata(args.metadata_file, metadata_records)

    reranked_results = sorted(
        reranked_results,
        key=lambda x: x["final_score"],
        reverse=True,
    )

    top_results = reranked_results[:args.top_k]

    if len(top_results) == 0:
        print("No results found.")
        print("Try:")
        print("python search.py --allow-different-class")
        return

    result_lines = []
    html_items = []

    result_lines.append(f"Query image: {query_image_path}")
    result_lines.append(f"Query detected object: {query_object}")
    result_lines.append(f"Query YOLO confidence: {round(query_yolo_conf * 100, 2)}%")
    result_lines.append(f"Query masked number: {query_masked_number}")
    result_lines.append(f"Query number type: {query_number_type}")
    result_lines.append("")
    result_lines.append("Top results:")
    result_lines.append("")

    print("\nTop Matching Results:\n")

    for rank, item in enumerate(top_results, start=1):
        metadata = item["metadata"]

        image_path = metadata["image_path"]
        detected_object = metadata.get("detected_object")
        masked_number = metadata.get("masked_number")
        number_type = metadata.get("number_type")

        visual_percentage = score_to_percentage(item["visual_score"])
        color_percentage = score_to_percentage(item["color_score"])
        final_percentage = score_to_percentage(item["final_score"])

        matched_crop = crop_from_metadata(image_path, metadata)

        original_name = Path(image_path).name
        output_filename = safe_filename(
            f"match_{rank}_{final_percentage}_percent_{original_name}"
        )

        output_path = result_dir / output_filename

        label_lines = [
            f"Rank #{rank}",
            f"Final match: {final_percentage}%",
            f"Visual match: {visual_percentage}%",
            f"Color match: {color_percentage}%",
            f"Detected: {detected_object}",
        ]

        if masked_number:
            label_lines.append(f"Number: {masked_number}")
            label_lines.append(f"Type: {number_type}")

        save_labeled_image(matched_crop, output_path, label_lines)

        print(f"Rank {rank}")
        print("Image:", image_path)
        print("Detected:", detected_object)
        print("Visual Similarity:", visual_percentage, "%")
        print("Color Similarity:", color_percentage, "%")
        print("Final Match:", final_percentage, "%")

        if masked_number:
            print("Number:", masked_number)
            print("Number Type:", number_type)

        print("Saved:", output_path)
        print()

        result_lines.append(f"Rank {rank}")
        result_lines.append(f"Image: {image_path}")
        result_lines.append(f"Detected: {detected_object}")
        result_lines.append(f"Visual Similarity: {visual_percentage}%")
        result_lines.append(f"Color Similarity: {color_percentage}%")
        result_lines.append(f"Final Match: {final_percentage}%")
        result_lines.append(f"Number: {masked_number}")
        result_lines.append(f"Number Type: {number_type}")
        result_lines.append(f"Saved: {output_path}")
        result_lines.append("")

        html_items.append(f"""
        <div style="border:1px solid #ccc; padding:15px; margin-bottom:25px;">
            <h2>Rank {rank} - Final Match: {final_percentage}%</h2>
            <p><b>Visual Similarity:</b> {visual_percentage}%</p>
            <p><b>Color Similarity:</b> {color_percentage}%</p>
            <p><b>Detected:</b> {html.escape(str(detected_object))}</p>
            <p><b>Number:</b> {html.escape(str(masked_number))}</p>
            <p><b>Number Type:</b> {html.escape(str(number_type))}</p>
            <p>{html.escape(str(image_path))}</p>
            <img src="{html.escape(output_filename)}" width="400">
        </div>
        """)

    txt_path = result_dir / "results.txt"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(result_lines))

    html_path = result_dir / "results.html"

    html_content = f"""
    <html>
    <head>
        <title>Lost and Found AI Results</title>
    </head>
    <body style="font-family:Arial; padding:20px;">
        <h1>AI-Powered Lost and Found Matching Results</h1>

        <h2>Query Image</h2>
        <p><b>Image:</b> {html.escape(str(query_image_path))}</p>
        <p><b>Detected Object:</b> {html.escape(str(query_object))}</p>
        <p><b>YOLO Confidence:</b> {round(query_yolo_conf * 100, 2)}%</p>
        <p><b>Number:</b> {html.escape(str(query_masked_number))}</p>
        <p><b>Number Type:</b> {html.escape(str(query_number_type))}</p>
        <img src="query_cropped.jpg" width="400">

        <hr>

        <h2>Top Matches</h2>
        {''.join(html_items)}
    </body>
    </html>
    """

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print("Results saved successfully.")
    print("Open this file:")
    print(html_path)


if __name__ == "__main__":
    main()