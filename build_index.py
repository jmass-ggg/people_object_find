import argparse
import os
import pickle
from pathlib import Path

import faiss
from tqdm import tqdm

from sensitive_ocr import SensitiveOCR
from vision_core import (
    VEHICLE_CLASSES,
    VisionCore,
    get_all_image_paths,
    get_color_histogram_from_pil,
)


DEFAULT_DATASET_DIR = "dataset"
DEFAULT_INDEX_FILE = "image_index.faiss"
DEFAULT_METADATA_FILE = "image_metadata.pkl"
DEFAULT_PATHS_FILE = "image_paths.pkl"


def load_existing_index(index_file, metadata_file, paths_file):
    if not (
        Path(index_file).exists()
        and Path(metadata_file).exists()
        and Path(paths_file).exists()
    ):
        return None, [], []

    print("Existing index found. Resuming...")

    index = faiss.read_index(index_file)

    with open(metadata_file, "rb") as f:
        metadata_records = pickle.load(f)

    with open(paths_file, "rb") as f:
        image_paths_saved = pickle.load(f)

    if index.ntotal != len(metadata_records):
        raise RuntimeError(
            "Index and metadata count mismatch. Delete old files and rebuild."
        )

    return index, metadata_records, image_paths_saved


def save_index_outputs(index, metadata_records, image_paths_saved, index_file, metadata_file, paths_file):
    if index is None or index.ntotal == 0:
        print("Nothing to save.")
        return

    faiss.write_index(index, index_file)

    with open(metadata_file, "wb") as f:
        pickle.dump(metadata_records, f)

    with open(paths_file, "wb") as f:
        pickle.dump(image_paths_saved, f)

    print("\nSaved index.")
    print("Total images indexed:", len(metadata_records))
    print("Saved:", index_file)
    print("Saved:", metadata_file)
    print("Saved:", paths_file)


def delete_existing_files(index_file, metadata_file, paths_file):
    for file_path in [index_file, metadata_file, paths_file]:
        if Path(file_path).exists():
            os.remove(file_path)
            print("Deleted:", file_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build FAISS image index using YOLO + CLIP."
    )

    parser.add_argument("--dataset", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--index-file", default=DEFAULT_INDEX_FILE)
    parser.add_argument("--metadata-file", default=DEFAULT_METADATA_FILE)
    parser.add_argument("--paths-file", default=DEFAULT_PATHS_FILE)

    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--yolo-confidence", type=float, default=0.25)

    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete old index files and build from scratch.",
    )

    parser.add_argument(
        "--enable-ocr",
        action="store_true",
        help="Run OCR during indexing. Not recommended on CPU.",
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.fresh:
        delete_existing_files(
            args.index_file,
            args.metadata_file,
            args.paths_file,
        )

    image_paths = get_all_image_paths(args.dataset)

    if len(image_paths) == 0:
        print("No images found inside dataset folder.")
        return

    print(f"Found {len(image_paths)} images.")

    index, metadata_records, image_paths_saved = load_existing_index(
        args.index_file,
        args.metadata_file,
        args.paths_file,
    )

    already_done = set(str(path) for path in image_paths_saved)

    pending_paths = [
        path for path in image_paths
        if str(path) not in already_done
    ]

    if len(pending_paths) == 0:
        print("All images are already indexed.")
        return

    print("Pending images:", len(pending_paths))

    vision = VisionCore(
        clip_model_name=args.clip_model,
        yolo_model_name=args.yolo_model,
        yolo_confidence=args.yolo_confidence,
        prefer_non_person=True,
    )

    ocr_reader = SensitiveOCR(
        enabled=args.enable_ocr,
    )

    processed_since_save = 0

    try:
        for image_path in tqdm(pending_paths):
            try:
                cropped_image, detection_info = vision.detect_and_crop_primary_object(image_path)

                embedding = vision.get_clip_embedding_from_pil(cropped_image)

                if index is None:
                    dimension = embedding.shape[1]
                    index = faiss.IndexFlatIP(dimension)

                if index.d != embedding.shape[1]:
                    raise RuntimeError(
                        f"Embedding dimension mismatch for {image_path}"
                    )

                detected_object = detection_info["detected_object"]

                color_histogram = get_color_histogram_from_pil(cropped_image)

                number_hash = None
                masked_number = None
                number_type = None
                number_confidence = 0.0

                if args.enable_ocr:
                    purpose = "plate" if detected_object in VEHICLE_CLASSES else "document"

                    ocr_result = ocr_reader.extract_from_pil(
                        cropped_image,
                        purpose=purpose,
                    )

                    number_hash = ocr_result["number_hash"]
                    masked_number = ocr_result["masked_number"]
                    number_type = ocr_result["number_type"]
                    number_confidence = ocr_result["confidence"]

                metadata = {
                    "image_path": str(image_path),
                    "detected_object": detected_object,
                    "yolo_confidence": detection_info["yolo_confidence"],
                    "used_original": detection_info["used_original"],
                    "bbox": detection_info["bbox"],
                    "color_histogram": color_histogram.tolist(),
                    "number_hash": number_hash,
                    "masked_number": masked_number,
                    "number_type": number_type,
                    "number_confidence": number_confidence,
                }

                index.add(embedding)
                metadata_records.append(metadata)
                image_paths_saved.append(str(image_path))

                processed_since_save += 1

                if processed_since_save >= args.checkpoint_every:
                    save_index_outputs(
                        index,
                        metadata_records,
                        image_paths_saved,
                        args.index_file,
                        args.metadata_file,
                        args.paths_file,
                    )
                    processed_since_save = 0

            except Exception as e:
                print("\nSkipping image:", image_path)
                print("Reason:", e)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving progress...")

    save_index_outputs(
        index,
        metadata_records,
        image_paths_saved,
        args.index_file,
        args.metadata_file,
        args.paths_file,
    )

    class_count = {}

    for metadata in metadata_records:
        detected_object = metadata.get("detected_object", "unknown")
        class_count[detected_object] = class_count.get(detected_object, 0) + 1

    print("\nDetected object summary:")
    for class_name, count in sorted(class_count.items(), key=lambda x: x[1], reverse=True):
        print(f"{class_name}: {count}")


if __name__ == "__main__":
    main()