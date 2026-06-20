import re
import cv2
import hashlib
import numpy as np
from PIL import Image


class PlateOCR:
    """
    General sensitive-number OCR.

    Can detect:
    - license plate numbers
    - passport numbers
    - national ID numbers
    - general printed numbers

    It tries PaddleOCR first.
    If PaddleOCR is not installed, it falls back to EasyOCR.
    """

    def __init__(self, enabled=True, gpu=False, lang="en"):
        self.enabled = enabled
        self.reader = None
        self.backend = None
        self.lang = lang

        if not enabled:
            print("Sensitive number OCR disabled.")
            return

        if self._load_paddleocr(gpu=gpu, lang=lang):
            return

        if self._load_easyocr(gpu=gpu):
            return

        print("No OCR backend could be loaded.")
        self.enabled = False

    def _load_paddleocr(self, gpu=False, lang="en"):
        try:
            from paddleocr import PaddleOCR

            print("Loading PaddleOCR...")

            try:
                self.reader = PaddleOCR(
                    use_angle_cls=True,
                    lang=lang,
                    use_gpu=gpu,
                    show_log=False,
                )
            except TypeError:
                try:
                    self.reader = PaddleOCR(
                        use_angle_cls=True,
                        lang=lang,
                    )
                except TypeError:
                    self.reader = PaddleOCR(lang=lang)

            self.backend = "paddleocr"
            print("PaddleOCR loaded successfully.")
            return True

        except Exception as e:
            print("PaddleOCR could not be loaded.")
            print("Reason:", e)
            return False

    def _load_easyocr(self, gpu=False):
        try:
            import easyocr

            print("Loading EasyOCR...")
            self.reader = easyocr.Reader(["en"], gpu=gpu)
            self.backend = "easyocr"
            print("EasyOCR loaded successfully.")
            return True

        except Exception as e:
            print("EasyOCR could not be loaded.")
            print("Reason:", e)
            return False

    def pil_to_rgb(self, image):
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
            return np.array(image)

        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        return image

    def pil_to_bgr(self, image):
        rgb = self.pil_to_rgb(image)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def resize_max_width(self, image, max_width=900):
        h, w = image.shape[:2]

        if w <= max_width:
            return image

        scale = max_width / w
        new_w = int(w * scale)
        new_h = int(h * scale)

        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def normalize_text(self, text):
        if text is None:
            return ""

        text = str(text).upper()
        text = text.replace("|", "I")
        text = text.replace(" ", "")
        text = text.replace("-", "")
        text = text.replace("_", "")
        text = text.replace(".", "")
        text = text.replace(":", "")
        text = text.replace("/", "")
        text = text.replace("\\", "")

        return text

    def alphanumeric_only(self, text):
        text = self.normalize_text(text)
        return re.sub(r"[^A-Z0-9]", "", text)

    def mask_value(self, value):
        if value is None:
            return None

        value = str(value)

        if len(value) <= 4:
            return value[0] + "***"

        if len(value) <= 8:
            return value[:2] + "****" + value[-1:]

        return value[:4] + "****" + value[-2:]

    def hash_value(self, value):
        if not value:
            return None

        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def is_possible_plate(self, text):
        if text is None:
            return False

        text = self.alphanumeric_only(text)

        if len(text) < 4 or len(text) > 15:
            return False

        has_digit = any(ch.isdigit() for ch in text)
        has_letter = any(ch.isalpha() for ch in text)

        return has_digit and has_letter

    def is_possible_passport(self, text):
        text = self.alphanumeric_only(text)

        # Common passport format: one letter + 6 to 9 digits
        if re.fullmatch(r"[A-Z][0-9]{6,9}", text):
            return True

        # Some passports are 2 letters + 6 to 8 digits
        if re.fullmatch(r"[A-Z]{2}[0-9]{6,8}", text):
            return True

        return False

    def is_possible_national_id(self, text):
        text = self.alphanumeric_only(text)

        # General national ID style: mostly digits, 6-20 chars
        if re.fullmatch(r"[0-9]{6,20}", text):
            return True

        # Some IDs are alphanumeric
        if re.fullmatch(r"[A-Z0-9]{8,20}", text):
            has_digit = any(ch.isdigit() for ch in text)
            return has_digit

        return False

    def generate_vehicle_plate_crops(self, image_bgr):
        image_bgr = self.resize_max_width(image_bgr, max_width=900)

        h, w = image_bgr.shape[:2]

        crops = [
            (
                "plate_center",
                image_bgr[
                    int(h * 0.55):int(h * 0.92),
                    int(w * 0.15):int(w * 0.85),
                ],
            ),
            (
                "bumper_wide",
                image_bgr[
                    int(h * 0.48):int(h * 0.95),
                    int(w * 0.05):int(w * 0.95),
                ],
            ),
            (
                "bottom_half",
                image_bgr[
                    int(h * 0.40):h,
                    0:w,
                ],
            ),
            (
                "full_vehicle",
                image_bgr,
            ),
        ]

        return self._valid_crops(crops)

    def generate_document_crops(self, image_bgr):
        image_bgr = self.resize_max_width(image_bgr, max_width=1000)

        h, w = image_bgr.shape[:2]

        crops = [
            ("full_document", image_bgr),
            (
                "top_half",
                image_bgr[
                    0:int(h * 0.55),
                    0:w,
                ],
            ),
            (
                "middle",
                image_bgr[
                    int(h * 0.20):int(h * 0.80),
                    0:w,
                ],
            ),
            (
                "bottom_half",
                image_bgr[
                    int(h * 0.45):h,
                    0:w,
                ],
            ),
            (
                "right_side",
                image_bgr[
                    0:h,
                    int(w * 0.35):w,
                ],
            ),
        ]

        return self._valid_crops(crops)

    def _valid_crops(self, crops):
        valid = []

        for name, crop in crops:
            if crop is not None and crop.shape[0] > 25 and crop.shape[1] > 60:
                valid.append((name, crop))

        return valid

    def preprocess_crop_versions(self, crop_bgr, purpose="auto"):
        crop_bgr = self.resize_max_width(crop_bgr, max_width=700)

        versions = []

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        versions.append(("color", rgb))

        # For documents, color OCR is usually enough and faster.
        if purpose == "document":
            return versions

        enlarged = cv2.resize(
            crop_bgr,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC,
        )

        gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 7, 11, 11)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )

        contrast = clahe.apply(gray)

        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ])

        sharpened = cv2.filter2D(contrast, -1, kernel)

        versions.append(("sharpened", sharpened))

        return versions

    def run_ocr_on_image(self, image):
        if not self.enabled or self.reader is None:
            return []

        if self.backend == "paddleocr":
            return self._run_paddleocr(image)

        if self.backend == "easyocr":
            return self._run_easyocr(image)

        return []

    def _run_paddleocr(self, image):
        try:
            try:
                result = self.reader.ocr(image, cls=True)
            except TypeError:
                result = self.reader.ocr(image)

            return self._parse_paddleocr_result(result)

        except Exception as e:
            print("PaddleOCR read failed:", e)
            return []

    def _parse_paddleocr_result(self, result):
        items = []

        def parse_obj(obj):
            if obj is None:
                return

            if isinstance(obj, dict):
                texts = (
                    obj.get("rec_texts")
                    or obj.get("texts")
                    or obj.get("text")
                    or []
                )

                scores = (
                    obj.get("rec_scores")
                    or obj.get("scores")
                    or obj.get("confidence")
                    or []
                )

                boxes = (
                    obj.get("dt_polys")
                    or obj.get("boxes")
                    or obj.get("bbox")
                    or []
                )

                if isinstance(texts, str):
                    texts = [texts]

                if isinstance(scores, (float, int)):
                    scores = [scores]

                if len(scores) == 0:
                    scores = [1.0] * len(texts)

                if len(boxes) == 0:
                    boxes = [None] * len(texts)

                for i, text in enumerate(texts):
                    score = scores[i] if i < len(scores) else 1.0
                    box = boxes[i] if i < len(boxes) else None

                    items.append({
                        "bbox": box,
                        "text": str(text),
                        "confidence": float(score),
                    })

                return

            if isinstance(obj, (list, tuple)):
                # Classic PaddleOCR format:
                # [box, (text, confidence)]
                if len(obj) >= 2:
                    possible_text_score = obj[1]

                    if (
                        isinstance(possible_text_score, (list, tuple))
                        and len(possible_text_score) >= 2
                        and isinstance(possible_text_score[0], str)
                    ):
                        items.append({
                            "bbox": obj[0],
                            "text": str(possible_text_score[0]),
                            "confidence": float(possible_text_score[1]),
                        })
                        return

                for child in obj:
                    parse_obj(child)

        parse_obj(result)
        return items

    def _run_easyocr(self, image):
        allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"

        try:
            results = self.reader.readtext(
                image,
                detail=1,
                paragraph=False,
                allowlist=allowlist,
                decoder="greedy",
                batch_size=1,
                workers=0,
                text_threshold=0.25,
                low_text=0.15,
                link_threshold=0.15,
                canvas_size=1200,
                mag_ratio=1.0,
            )

            items = []

            for bbox, text, confidence in results:
                items.append({
                    "bbox": bbox,
                    "text": str(text),
                    "confidence": float(confidence),
                })

            return items

        except Exception as e:
            print("EasyOCR read failed:", e)
            return []

    def extract_candidates_from_text(self, raw_text, confidence, purpose="auto"):
        candidates = []

        if raw_text is None:
            return candidates

        raw_text = str(raw_text).upper()

        spaced = re.sub(r"[^A-Z0-9<]+", " ", raw_text)
        compact = self.alphanumeric_only(raw_text)

        if not compact:
            return candidates

        # MRZ lines, common on passports
        mrz_like = re.findall(r"[A-Z0-9<]{15,}", raw_text)
        for item in mrz_like:
            clean_mrz = item.replace(" ", "")

            if "<" in clean_mrz:
                value = re.sub(r"[^A-Z0-9]", "", clean_mrz[:12])

                if len(value) >= 6:
                    candidates.append({
                        "value": value,
                        "type": "mrz",
                        "confidence": confidence,
                        "score": confidence + 0.70,
                        "raw_text": raw_text,
                    })

        # Passport number
        passport_matches = re.findall(r"\b[A-Z]{1,2}[0-9]{6,9}\b", spaced)
        for value in passport_matches:
            value = self.alphanumeric_only(value)

            if self.is_possible_passport(value):
                candidates.append({
                    "value": value,
                    "type": "passport_number",
                    "confidence": confidence,
                    "score": confidence + 0.80,
                    "raw_text": raw_text,
                })

        # National ID / long number
        digit_matches = re.findall(r"\b[0-9]{6,20}\b", spaced)
        for value in digit_matches:
            value = self.alphanumeric_only(value)

            if self.is_possible_national_id(value):
                candidates.append({
                    "value": value,
                    "type": "national_id_number",
                    "confidence": confidence,
                    "score": confidence + 0.65,
                    "raw_text": raw_text,
                })

        # License plate
        if self.is_possible_plate(compact):
            plate_bonus = 0.0

            # Example: MH14BN7077
            if re.fullmatch(r"[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}", compact):
                plate_bonus += 0.60

            if 5 <= len(compact) <= 12:
                plate_bonus += 0.25

            candidates.append({
                "value": compact,
                "type": "license_plate",
                "confidence": confidence,
                "score": confidence + plate_bonus,
                "raw_text": raw_text,
            })

        # Generic number fallback
        if compact.isdigit() and 4 <= len(compact) <= 20:
            candidates.append({
                "value": compact,
                "type": "number",
                "confidence": confidence,
                "score": confidence + 0.20,
                "raw_text": raw_text,
            })

        # Purpose-based boost
        for candidate in candidates:
            if purpose == "plate" and candidate["type"] == "license_plate":
                candidate["score"] += 0.50

            if purpose == "document" and candidate["type"] in {
                "passport_number",
                "national_id_number",
                "mrz",
            }:
                candidate["score"] += 0.50

        return candidates

    def extract_detailed_from_pil(self, image, purpose="auto"):
        """
        Returns:
        {
            "number": str or None,
            "masked_number": str or None,
            "number_hash": str or None,
            "number_type": str or None,
            "confidence": float,
            "all_candidates": list
        }
        """

        if not self.enabled or self.reader is None:
            return {
                "number": None,
                "masked_number": None,
                "number_hash": None,
                "number_type": None,
                "confidence": 0.0,
                "all_candidates": [],
            }

        try:
            image_bgr = self.pil_to_bgr(image)

            if purpose == "plate":
                candidate_crops = self.generate_vehicle_plate_crops(image_bgr)
            elif purpose == "document":
                candidate_crops = self.generate_document_crops(image_bgr)
            else:
                candidate_crops = (
                    self.generate_vehicle_plate_crops(image_bgr)
                    + self.generate_document_crops(image_bgr)
                )

            all_candidates = []

            for crop_name, crop_bgr in candidate_crops:
                versions = self.preprocess_crop_versions(
                    crop_bgr,
                    purpose=purpose,
                )

                for version_name, processed_image in versions:
                    ocr_items = self.run_ocr_on_image(processed_image)

                    for item in ocr_items:
                        text = item.get("text")
                        confidence = float(item.get("confidence", 0.0))

                        candidates = self.extract_candidates_from_text(
                            raw_text=text,
                            confidence=confidence,
                            purpose=purpose,
                        )

                        for candidate in candidates:
                            candidate["crop"] = crop_name
                            candidate["version"] = version_name
                            all_candidates.append(candidate)

            if len(all_candidates) == 0:
                return {
                    "number": None,
                    "masked_number": None,
                    "number_hash": None,
                    "number_type": None,
                    "confidence": 0.0,
                    "all_candidates": [],
                }

            best = max(all_candidates, key=lambda x: x["score"])

            value = best["value"]

            return {
                "number": value,
                "masked_number": self.mask_value(value),
                "number_hash": self.hash_value(value),
                "number_type": best["type"],
                "confidence": best["confidence"],
                "all_candidates": all_candidates,
            }

        except Exception as e:
            print("Sensitive number OCR failed:", e)

            return {
                "number": None,
                "masked_number": None,
                "number_hash": None,
                "number_type": None,
                "confidence": 0.0,
                "all_candidates": [],
            }

    def extract_from_pil(self, image, purpose="auto"):
        """
        Keeps compatibility with your old code.

        Old return:
        plate_number, masked_plate, plate_confidence

        New behavior:
        number, masked_number, confidence
        """

        result = self.extract_detailed_from_pil(
            image=image,
            purpose=purpose,
        )

        return (
            result["number"],
            result["masked_number"],
            result["confidence"],
        )