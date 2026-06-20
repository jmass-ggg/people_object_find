from pathlib import Path
from uuid import uuid4

from search import (
    calculate_final_score,
    load_index_and_metadata,
)

from vision_core import (
    VisionCore,
    crop_from_metadata,
    get_color_histogram_from_pil,
    histogram_similarity,
    safe_filename,
    save_labeled_image,
    score_to_percentage,
)


class BlockedQueryError(Exception):
    pass


class ImageSearchService:
    def __init__(
        self,
        index_file="image_index.faiss",
        metadata_file="image_metadata.pkl",
        result_root="result/api",
        clip_model_name="ViT-B/32",
        yolo_model_name="yolov8n.pt",
        yolo_confidence=0.25,
        top_k=10,
        rerank_limit=100,
        allow_different_class=False,
    ):
        self.index, self.metadata_records = load_index_and_metadata(
            index_file,
            metadata_file,
        )

        self.result_root = Path(result_root)
        self.result_root.mkdir(parents=True, exist_ok=True)

        self.top_k = top_k
        self.rerank_limit = rerank_limit
        self.allow_different_class = allow_different_class

        print("Loading CLIP + YOLO once for FastAPI...")
        self.vision = VisionCore(
            clip_model_name=clip_model_name,
            yolo_model_name=yolo_model_name,
            yolo_confidence=yolo_confidence,
            prefer_non_person=True,
        )

    def search(self, query_image_path, top_k=None):
        top_k = top_k or self.top_k
        request_id = uuid4().hex

        request_dir = self.result_root / request_id
        request_dir.mkdir(parents=True, exist_ok=True)

        query_crop, query_info = self.vision.detect_and_crop_primary_object(
            query_image_path
        )

        query_object = query_info["detected_object"]
        query_yolo_confidence = query_info["yolo_confidence"]

        if query_object == "person":
            raise BlockedQueryError(
                "Person/face matching is disabled. Upload an object image such as a bag, phone, helmet, key, laptop, bottle, wallet, or similar item."
            )

        query_embedding = self.vision.get_clip_embedding_from_pil(query_crop)
        query_color_histogram = get_color_histogram_from_pil(query_crop)

        query_crop_path = request_dir / "query_cropped.jpg"

        save_labeled_image(
            query_crop,
            query_crop_path,
            [
                f"Detected: {query_object}",
                f"YOLO confidence: {round(query_yolo_confidence * 100, 2)}%",
            ],
        )

        search_k = min(self.rerank_limit, self.index.ntotal)

        scores, indices = self.index.search(
            query_embedding,
            search_k,
        )

        reranked_results = []

        for raw_rank, idx in enumerate(indices[0]):
            if idx == -1:
                continue

            metadata = self.metadata_records[idx]
            candidate_object = metadata.get("detected_object", "unknown")

            if candidate_object == "person":
                continue

            if not self.allow_different_class and query_object != "original":
                if candidate_object != query_object:
                    continue

            visual_score = float(scores[0][raw_rank])

            candidate_color_histogram = metadata.get("color_histogram")

            color_score = histogram_similarity(
                query_color_histogram,
                candidate_color_histogram,
            )

            final_score = calculate_final_score(
                visual_score=visual_score,
                color_score=color_score,
                query_number_hash=None,
                candidate_number_hash=None,
            )

            reranked_results.append(
                {
                    "metadata": metadata,
                    "visual_score": visual_score,
                    "color_score": color_score,
                    "final_score": final_score,
                }
            )

        reranked_results = sorted(
            reranked_results,
            key=lambda item: item["final_score"],
            reverse=True,
        )

        top_results = reranked_results[:top_k]

        results = []

        for rank, item in enumerate(top_results, start=1):
            metadata = item["metadata"]

            image_path = metadata["image_path"]
            detected_object = metadata.get("detected_object", "unknown")

            visual_percentage = score_to_percentage(item["visual_score"])
            color_percentage = score_to_percentage(item["color_score"])
            final_percentage = score_to_percentage(item["final_score"])

            matched_crop = crop_from_metadata(
                image_path,
                metadata,
            )

            original_name = Path(image_path).name

            output_filename = safe_filename(
                f"match_{rank}_{final_percentage}_percent_{original_name}"
            )

            output_path = request_dir / output_filename

            save_labeled_image(
                matched_crop,
                output_path,
                [
                    f"Rank #{rank}",
                    f"Final match: {final_percentage}%",
                    f"Visual match: {visual_percentage}%",
                    f"Color match: {color_percentage}%",
                    f"Detected: {detected_object}",
                ],
            )

            results.append(
                {
                    "rank": rank,
                    "image_path": image_path,
                    "detected_object": detected_object,
                    "visual_percentage": visual_percentage,
                    "color_percentage": color_percentage,
                    "final_percentage": final_percentage,
                    "image_url": f"/result/api/{request_id}/{output_filename}",
                }
            )

        return {
            "blocked": False,
            "request_id": request_id,
            "query": {
                "image_path": str(query_image_path),
                "detected_object": query_object,
                "yolo_confidence": round(query_yolo_confidence * 100, 2),
                "image_url": f"/result/api/{request_id}/query_cropped.jpg",
            },
            "results": results,
        }
