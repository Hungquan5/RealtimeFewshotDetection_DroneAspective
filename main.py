# ======================================================================
# IMPORTS
# ======================================================================
import numpy as np
import cv2
import os
import re
import json
import random
from typing import List, Optional, Tuple, Dict
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from skimage.feature import hog
from scipy.spatial.distance import euclidean
import math
from sklearn.cluster import KMeans # Import for K-Means

# Import for PDE-based analysis
from ultralytics import YOLOE, YOLO
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor

# +++ NEW IMPORTS FOR MOBILECLIP2 +++
# You will need to install open_clip_torch: pip install open_clip_torch
import open_clip
from PIL import Image

# ======================================================================
# YOLO-E PIPELINE CLASS (CORRECTED INTEGRATION)
# ======================================================================

class YOLOEPipeline:
    """
    A comprehensive pipeline for object detection and tracking using a hybrid, two-stage prompt approach.
    This version uses YOLOE with VPE for detection and MobileCLIP2 for similarity-based verification.
    """
    def __init__(self,
                 yoloe_model_path: str = "yoloe-11s-seg.pt",
                 yolov8_model_path: str = "yolov8n.pt",
                 clip_model_name: str = "MobileCLIP2-S0",
                 clip_pretrained: str = "dfndr2b",
                 seed: Optional[int] = None):
        """Initialize the pipeline with YOLOE, YOLOv8, and MobileCLIP2 models."""
        self.yoloe_model_path = yoloe_model_path
        self.yolov8_model_path = yolov8_model_path
        self.seed = seed
        if self.seed is not None:
            self._set_seed(self.seed)

        print("Initializing models...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.model = YOLOE(self.yoloe_model_path)
        self.detector = YOLO(self.yolov8_model_path)

        # +++ ADDED: Initialize MobileCLIP2 model for verification +++
        print(f"Initializing MobileCLIP2 model: {clip_model_name}...")
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            clip_model_name,
            pretrained=clip_pretrained
        )
        self.clip_model.to(self.device)
        self.clip_model.eval() # Set to evaluation mode
        print("Models initialized successfully.")

    @staticmethod
    def _set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        cv2.setRNGSeed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print(f"Random seed set to {seed}")

    @staticmethod
    def _natural_key(s: str):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

    def _group_consecutive_detections(self, detections: List[Dict]) -> List[Dict]:
        if not detections: return []
        detections.sort(key=lambda d: d["frame"])
        grouped_detections, current_group = [], [detections[0]]
        for i in range(1, len(detections)):
            if detections[i]["frame"] == detections[i-1]["frame"] + 1:
                current_group.append(detections[i])
            else:
                grouped_detections.append({"bboxes": current_group})
                current_group = [detections[i]]
        if current_group: grouped_detections.append({"bboxes": current_group})
        return grouped_detections

    def extract_class_name_from_folder(self, folder_name: str) -> str:
        return re.sub(r'_\d+$', '', folder_name)

    @staticmethod
    def _crop_object(
            image: np.ndarray,
            bbox: Tuple[float, float, float, float],
            mask: Optional[np.ndarray] = None
        ) -> Optional[np.ndarray]:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(int(round(x1)), w - 1))
        y1 = max(0, min(int(round(y1)), h - 1))
        x2 = max(0, min(int(round(x2)), w))
        y2 = max(0, min(int(round(y2)), h))
        if x1 >= x2 or y1 >= y2: return None
        crop = image[y1:y2, x1:x2]
        if mask is None: return crop
        full_mask = mask
        if full_mask.ndim == 3: full_mask = full_mask.squeeze()
        mask_crop = full_mask[y1:y2, x1:x2]
        if mask_crop.dtype != np.uint8:
            binary_mask = (mask_crop > 0.4).astype(np.uint8)
        else:
            binary_mask = (mask_crop > 0).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(binary_mask, kernel, iterations=2)
        alpha_channel = dilated_mask * 255
        if alpha_channel.shape[:2] != crop.shape[:2]:
            alpha_channel = cv2.resize(alpha_channel, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
        BGRa_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
        BGRa_crop[:, :, 3] = alpha_channel
        return BGRa_crop

    @staticmethod
    def _clip_bbox_to_frame(bbox: Tuple[int, int, int, int], frame_width: int, frame_height: int) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        return max(0, x1), max(0, y1), min(frame_width, x2), min(frame_height, y2)

    # --- RESTORED: This function is essential for YOLOE prompting ---
    def _extract_vpe(self, image: np.ndarray, bboxes: List[List[int]]) -> Optional[torch.Tensor]:
        """Extracts VPE from a full image using bounding box prompts."""
        self.model.predictor = None
        if not bboxes: return None
        temp_img_path = f"temp_frame_for_vpe_{random.randint(0, 99999)}.jpg"
        cv2.imwrite(temp_img_path, image)
        try:
            visual_prompts = {'bboxes': [np.array(bboxes)], 'cls': [np.array([0] * len(bboxes))]}
            self.model.predict(
                temp_img_path, prompts=visual_prompts, predictor=YOLOEVPSegPredictor,
                return_vpe=True, verbose=False
            )
            vpe = self.model.predictor.vpe
            self.model.predictor = None
        except Exception as e:
            print(f"Error during VPE extraction: {e}")
            vpe = None
        finally:
            if os.path.exists(temp_img_path): os.remove(temp_img_path)
        return vpe

    # +++ ADDED: New function to get MobileCLIP2 embeddings for verification +++
    def _extract_mobileclip2_embedding(self, image_crop: np.ndarray) -> Optional[torch.Tensor]:
        """Extracts image embeddings from an image crop using MobileCLIP2."""
        if image_crop is None or image_crop.size == 0: return None
        try:
            if image_crop.shape[2] == 4:
                image_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGRA2RGB)
            else:
                image_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_crop)
            image_tensor = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.clip_model.encode_image(image_tensor)
                embedding = F.normalize(embedding, p=2, dim=-1)
            return embedding.cpu()
        except Exception as e:
            print(f"Error during MobileCLIP2 embedding extraction: {e}")
            return None

    def _calculate_color_similarity(self, crop1: np.ndarray, crop2: np.ndarray, num_colors: int = 2) -> float:
        # This function remains unchanged from your original code.
        # ... (full implementation of _calculate_color_similarity)
        # --- CIEDE2000 implementation (as a nested function) ---
        def deltaE_ciede2000(L1, a1, b1, L2, a2, b2, kL=1.0, kC=1.0, kH=1.0) -> float:
            C1 = math.sqrt(a1 * a1 + b1 * b1)
            C2 = math.sqrt(a2 * a2 + b2 * b2)
            C_bar = 0.5 * (C1 + C2)
            C_bar7 = C_bar ** 7
            G = 0.5 * (1.0 - math.sqrt(C_bar7 / (C_bar7 + 25.0 ** 7 + 1e-12)))

            a1p = (1.0 + G) * a1
            a2p = (1.0 + G) * a2
            C1p = math.sqrt(a1p * a1p + b1 * b1)
            C2p = math.sqrt(a2p * a2p + b2 * b2)
            C_bar_p = 0.5 * (C1p + C2p)

            def _hp(ap, b):
                if ap == 0.0 and b == 0.0:
                    return 0.0
                h = math.degrees(math.atan2(b, ap))
                return h + 360.0 if h < 0.0 else h

            h1p = _hp(a1p, b1)
            h2p = _hp(a2p, b2)

            dLp = L2 - L1
            dCp = C2p - C1p

            if C1p * C2p == 0:
                dHp = 0.0
            else:
                dh = h2p - h1p
                if dh > 180.0:
                    dh -= 360.0
                elif dh < -180.0:
                    dh += 360.0
                dHp = 2.0 * math.sqrt(C1p * C2p) * math.sin(math.radians(dh / 2.0))

            L_bar_p = 0.5 * (L1 + L2)

            if C1p * C2p == 0:
                H_bar_p = h1p + h2p
            else:
                dh = abs(h1p - h2p)
                if dh > 180.0:
                    H_bar_p = 0.5 * (h1p + h2p + 360.0) if (h1p + h2p) < 360.0 else 0.5 * (h1p + h2p - 360.0)
                else:
                    H_bar_p = 0.5 * (h1p + h2p)

            T = (1.0 - 0.17 * math.cos(math.radians(H_bar_p - 30.0))
                 + 0.24 * math.cos(math.radians(2.0 * H_bar_p))
                 + 0.32 * math.cos(math.radians(3.0 * H_bar_p + 6.0))
                 - 0.20 * math.cos(math.radians(4.0 * H_bar_p - 63.0)))

            dtheta = 30.0 * math.exp(-(((H_bar_p - 275.0) / 25.0) ** 2))
            Rc = 2.0 * math.sqrt((C_bar_p ** 7) / (C_bar_p ** 7 + 25.0 ** 7 + 1e-12))

            Sl = 1.0 + (0.015 * ((L_bar_p - 50.0) ** 2)) / math.sqrt(20.0 + (L_bar_p - 50.0) ** 2)
            Sc = 1.0 + 0.045 * C_bar_p
            Sh = 1.0 + 0.015 * C_bar_p * T

            Rt = -math.sin(math.radians(2.0 * dtheta)) * Rc

            dE = math.sqrt((dLp / (kL * Sl)) ** 2
                           + (dCp / (kC * Sc)) ** 2
                           + (dHp / (kH * Sh)) ** 2
                           + Rt * (dCp / (kC * Sc)) * (dHp / (kH * Sh)))

            return float(dE)

        try:
            # --- Helper: extract dominant LAB colors + weights ---
            def get_dominant_colors(crop: np.ndarray):
                # Handle transparency
                if crop.shape[2] == 4:
                    mask = crop[:, :, 3] > 0
                    if not np.any(mask):
                        return None, None
                    bgr_pixels = crop[:, :, :3][mask]
                else:
                    bgr_pixels = crop.reshape(-1, 3)

                if bgr_pixels.shape[0] < num_colors:
                    return None, None

                # For performance, resize before clustering if many pixels
                max_pixels = 10000  # Process at most 100x100 pixels
                if bgr_pixels.shape[0] > max_pixels:
                    # Create a temporary flat image to resize
                    temp_img = bgr_pixels.reshape(1, -1, 3).astype(np.uint8)
                    scale = np.sqrt(max_pixels / bgr_pixels.shape[0])
                    small_img = cv2.resize(temp_img, (0, 0), fx=scale, fy=1.0, interpolation=cv2.INTER_AREA)
                    bgr_pixels = small_img.reshape(-1, 3)

                # Convert to LAB color space for more perceptually uniform clustering
                lab_pixels = cv2.cvtColor(bgr_pixels.reshape(1, -1, 3).astype(np.uint8),
                                         cv2.COLOR_BGR2LAB).reshape(-1, 3)

                # Perform K-Means clustering
                kmeans = KMeans(n_clusters=num_colors, random_state=self.seed, n_init='auto')
                kmeans.fit(lab_pixels)

                unique_labels, counts = np.unique(kmeans.labels_, return_counts=True)
                weights = counts / counts.sum()
                dominant_colors_lab = kmeans.cluster_centers_

                # Sort by weight (most dominant first)
                sorted_indices = np.argsort(weights)[::-1]
                return dominant_colors_lab[sorted_indices], weights[sorted_indices]

            dom_colors1, weights1 = get_dominant_colors(crop1)
            dom_colors2, weights2 = get_dominant_colors(crop2)

            if dom_colors1 is None or dom_colors2 is None:
                return 0.0

            # --- Helper: Calculate palette distance in one direction ---
            def calculate_palette_distance(colors_from, weights_from, colors_to):
                total_distance = 0.0

                for i in range(len(colors_from)):
                    color_from_lab_cv = colors_from[i]
                    weight_from = weights_from[i]

                    # Convert OpenCV LAB -> "true" CIE LAB
                    L1 = color_from_lab_cv[0] * (100.0 / 255.0)
                    a1 = color_from_lab_cv[1] - 128.0
                    b1 = color_from_lab_cv[2] - 128.0

                    # Find the closest color in the target palette
                    min_dist = float('inf')
                    for color_to_lab_cv in colors_to:
                        L2 = color_to_lab_cv[0] * (100.0 / 255.0)
                        a2 = color_to_lab_cv[1] - 128.0
                        b2 = color_to_lab_cv[2] - 128.0

                        dist = deltaE_ciede2000(L1, a1, b1, L2, a2, b2)
                        if dist < min_dist:
                            min_dist = dist

                    total_distance += weight_from * min_dist

                return total_distance

            # --- Helper: Calculate color coverage (for penalty) ---
            def calculate_coverage(colors_from, weights_from, colors_to, threshold=20.0):
                """
                Calculate how much of the source palette is well-represented in the target palette.
                Returns the sum of weights for colors that have a match below the threshold.
                """
                covered_weight = 0.0

                for i in range(len(colors_from)):
                    color_from_lab_cv = colors_from[i]
                    weight_from = weights_from[i]

                    # Convert OpenCV LAB -> "true" CIE LAB
                    L1 = color_from_lab_cv[0] * (100.0 / 255.0)
                    a1 = color_from_lab_cv[1] - 128.0
                    b1 = color_from_lab_cv[2] - 128.0

                    # Find the closest color in the target palette
                    min_dist = float('inf')
                    for color_to_lab_cv in colors_to:
                        L2 = color_to_lab_cv[0] * (100.0 / 255.0)
                        a2 = color_to_lab_cv[1] - 128.0
                        b2 = color_to_lab_cv[2] - 128.0

                        dist = deltaE_ciede2000(L1, a1, b1, L2, a2, b2)
                        if dist < min_dist:
                            min_dist = dist

                    # If the color is well-matched (below threshold), count its weight
                    if min_dist < threshold:
                        covered_weight += weight_from

                return covered_weight

            # --- Bidirectional matching: calculate distance in both directions ---
            dist_1to2 = calculate_palette_distance(dom_colors1, weights1, dom_colors2)
            dist_2to1 = calculate_palette_distance(dom_colors2, weights2, dom_colors1)

            # Use symmetric distance (average)
            total_distance = (dist_1to2 + dist_2to1) / 2.0

            # --- Calculate coverage in both directions ---
            coverage_threshold = 20.0  # Colors with deltaE < 20 are considered "matched"
            coverage1 = calculate_coverage(dom_colors1, weights1, dom_colors2, coverage_threshold)
            coverage2 = calculate_coverage(dom_colors2, weights2, dom_colors1, coverage_threshold)

            # Use the minimum coverage (stricter check)
            min_coverage = min(coverage1, coverage2)

            # Calculate coverage penalty (0 when fully covered, increases as coverage decreases)
            coverage_penalty = 1.0 - min_coverage

            # --- Map weighted average ΔE00 -> [0,1] similarity ---
            sigma = 15.0  # Tune based on how strictly you want to penalize color differences
            similarity = math.exp(-(total_distance ** 2) / (2.0 * sigma * sigma))

            # Apply coverage penalty (adjust the 0.5 factor to tune penalty strength)
            # 0.5 means missing colors can reduce similarity by up to 50%
            penalty_strength = 0.5
            similarity *= (1.0 - coverage_penalty * penalty_strength)

            return float(np.clip(similarity, 0.0, 1.0))

        except (cv2.error, IndexError, ValueError) as e:
            print(f"Error during K-Means color similarity calculation: {e}")
            return 0.0

    # --- MODIFIED: This function now uses MobileCLIP2 embeddings for similarity calculation ---
    def _calculate_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor, crop1: np.ndarray, crop2: np.ndarray,
                              embedding_weight: float = 0.7, color_weight: float = 0.3) -> float:
        """Computes a combined similarity score using MobileCLIP2 embedding and color features."""
        # MobileCLIP2 Embedding Similarity (Cosine Similarity)
        embedding_similarity = F.cosine_similarity(emb1, emb2).item()
        # Color Similarity
        color_similarity = self._calculate_color_similarity(crop1, crop2)
        # Weighted Combination
        combined_similarity = (embedding_weight * embedding_similarity + color_weight * color_similarity)
        return combined_similarity

    # --- MODIFIED: This function now extracts reference embeddings and crops for verification ---
    def extract_reference_data(self, object_images_dir: str, conf_threshold: float = 0.1) -> List[Tuple[torch.Tensor, np.ndarray]]:
        """Extracts MobileCLIP2 embeddings and crops from original sample images for ground-truth similarity check."""
        reference_data = []
        image_files = [f for f in os.listdir(object_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for img_file in sorted(image_files, key=self._natural_key):
            img_path = os.path.join(object_images_dir, img_file)
            img = cv2.imread(img_path)
            if img is None: continue
            results = self.detector.predict(img_path, conf=conf_threshold, verbose=False)
            result = results[0]
            if result.boxes:
                best_box_coords = result.boxes.xyxy[result.boxes.conf.argmax()].cpu().numpy()
                best_box = tuple(map(int, best_box_coords))
                crop = self._crop_object(img, best_box)
                if crop is not None:
                    embedding = self._extract_mobileclip2_embedding(crop)
                    if embedding is not None:
                         reference_data.append((embedding, crop))
        print(f"     Extracted {len(reference_data)} reference embeddings and crops for similarity comparison.")
        return reference_data

    # --- All the following helper methods from your original code are kept as they are ---
    def sample_video_frames(self, video_path: str, num_frames: int = 5) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): raise RuntimeError(f"Cannot open video: {video_path}")
        frames = []
        for _ in range(num_frames):
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)
        cap.release()
        return frames

    def jitter_brightness_contrast(self, patch: np.ndarray, jb: float = 0.35, jc: float = 0.2) -> np.ndarray:
        b_factor = 1.0 + random.uniform(-jb, jb)
        c_factor = 1.0 + random.uniform(-jc, jc)
        patch_f = (patch.astype(np.float32) - patch.mean()) * c_factor + patch.mean()
        patch_f = np.clip(patch_f * b_factor, 0, 255).astype(np.uint8)
        return patch_f

    def _score_detection(self, bbox: Tuple[int, int, int, int], confidence: float, area_weight: float, conf_weight: float) -> float:
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        return area_weight * min(1.0, area / (640 * 640)) + conf_weight * confidence

    def detect_and_extract_objects(self, object_images_dir: str, conf_threshold: float, top_k: int, area_weight: float, conf_weight: float) -> List[Tuple[np.ndarray, Tuple, float]]:
        """Detects objects and extracts the top-k scored CROPS for augmentation."""
        crops = []
        image_files = [f for f in os.listdir(object_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"     Found {len(image_files)} object images.")
        for img_file in sorted(image_files, key=self._natural_key):
            img_path = os.path.join(object_images_dir, img_file)
            img = cv2.imread(img_path)
            if img is None: continue
            results = self.detector.predict(img_path, save=False, conf=conf_threshold, verbose=False)
            result = results[0]
            if result.boxes:
                detections = [
                    {'bbox': tuple(map(int, box)), 'conf': conf, 'score': self._score_detection(tuple(map(int, box)), conf, area_weight, conf_weight)}
                    for box, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy())
                ]
                for det in sorted(detections, key=lambda x: x['score'], reverse=True)[:top_k]:
                    x1, y1, x2, y2 = det['bbox']
                    crop = img[y1:y2, x1:x2].copy()
                    if crop.size > 0: crops.append((crop, det['bbox'], det['conf']))
        print(f"     Extracted {len(crops)} top-quality object crops for augmentation.")
        return crops

    def create_augmented_backgrounds(self, video_path: str, object_crops: List, num_backgrounds: int, objects_per_bg: int, min_scale: float, max_scale: float, prefer_high_conf: bool, output_dir: Optional[str] = None) -> Tuple[List[np.ndarray], List[List[Tuple]]]:
        if not object_crops: raise ValueError("No object crops provided for augmentation.")
        bg_frames = self.sample_video_frames(video_path, num_backgrounds)
        if not bg_frames: raise RuntimeError("Failed to sample frames from the video.")
        augmented_images, all_bboxes = [], []
        weights = np.array([c[2] for c in object_crops])
        weights /= weights.sum() if weights.sum() > 0 else 1.0

        for bg_idx, bg in enumerate(bg_frames):
            H, W, _ = bg.shape
            bg_copy, frame_bboxes = bg.copy(), []
            for _ in range(min(objects_per_bg, len(object_crops))):
                crop_data = object_crops[np.random.choice(len(object_crops), p=weights) if prefer_high_conf else random.randint(0, len(object_crops) - 1)]
                crop = crop_data[0]
                obj = self.jitter_brightness_contrast(crop.copy())
                scale = random.uniform(min_scale, max_scale)
                tgt_w, tgt_h = max(8, int(round(W * scale))), max(8, int(round(obj.shape[0] * (max(8, int(round(W * scale))) / max(1, obj.shape[1])))))
                if tgt_w <= 0 or tgt_h <= 0: continue
                obj_small = cv2.resize(obj, (tgt_w, tgt_h), interpolation=cv2.INTER_AREA)
                x1, y1 = random.randint(0, max(0, W - tgt_w - 1)), random.randint(0, max(0, H - tgt_h - 1))
                if x1 + tgt_w > W or y1 + tgt_h > H: continue
                bg_copy[y1:y1+tgt_h, x1:x1+tgt_w] = obj_small
                frame_bboxes.append((x1, y1, x1+tgt_w, y1+tgt_h))
            augmented_images.append(bg_copy)
            all_bboxes.append(frame_bboxes)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                cv2.imwrite(os.path.join(output_dir, f"aug_bg_{bg_idx:02d}.jpg"), bg_copy)
        print(f"     Created {len(augmented_images)} augmented backgrounds.")
        return augmented_images, all_bboxes

    def _detect_and_process_all(self, frame: np.ndarray, conf_threshold: float) -> List[Dict]:
        results = self.model(frame, save=False, conf=conf_threshold, verbose=False, retina_masks=True)
        result = results[0]
        detections = []
        if result.boxes:
            for i in range(len(result.boxes)):
                box = result.boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, box)
                detections.append({
                    "xyxy": (x1, y1, x2, y2),
                    "xywh": (x1, y1, x2 - x1, y2 - y1),
                    "conf": float(result.boxes.conf[i].cpu().numpy()),
                    "mask": result.masks.data[i].cpu().numpy() if result.masks is not None else None,
                })
        return detections

    # --- MODIFIED: The core video processing logic is updated for the new verification flow ---
    def process_video(
        self, augmented_images: List[np.ndarray], bboxes_list: List[List[Tuple]], video_path: str, class_name: str,
        reference_data: List[Tuple[torch.Tensor, np.ndarray]], conf_threshold: float, tracker_reinit_interval: int,
        save_frames: bool, output_frames_dir: Optional[str], save_interval: int,
        similarity_neg_threshold_factor: float = 0.95,
        edge_proximity_threshold: int = 10,
        similarity_add_threshold: float = 0.55
    ) -> Tuple[List[Dict], Dict]:
        # --- Step 1: Create and set the initial VPE prompt for YOLOE (Restored) ---
        self.model.predictor = None
        initial_vpes = [vpe for aug_img, bboxes in zip(augmented_images, bboxes_list) if (vpe := self._extract_vpe(aug_img, bboxes)) is not None]
        if not initial_vpes: raise RuntimeError("Could not generate an initial VPE from augmented images.")
        initial_vpe = torch.mean(torch.cat(initial_vpes, dim=0), dim=0, keepdim=True)
        print(f"     Setting class name: '{class_name}' with initial averaged VPE.")
        self.model.set_classes([class_name], initial_vpe)
        self.model.predictor = None

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        tracker, detections, frames_since_detection = None, [], 0
        highest_similarity_score = 0.2 # Starting threshold
        dynamic_reference_data = list(reference_data) # Make a copy to modify

        if save_frames and output_frames_dir: os.makedirs(output_frames_dir, exist_ok=True)
        print("     Starting video processing with MobileCLIP2 verification...")

        for frame_idx in range(total_frames):
            ok, frame = cap.read()
            if not ok: break

            bbox_for_drawing = None
            detection_status = "Unknown"

            bbox_from_tracker = None
            if tracker is not None:
                success, box = tracker.update(frame)
                if success:
                    x, y, w, h = [int(v) for v in box]
                    clipped_box = self._clip_bbox_to_frame((x, y, x + w, y + h), frame_w, frame_h)
                    if (clipped_box[2] - clipped_box[0]) > 0: bbox_from_tracker = clipped_box
                else: tracker = None

            run_detector = (tracker is None) or (frames_since_detection >= tracker_reinit_interval)

            if run_detector:
                detection_status = "Detecting"
                is_near_edge = False
                if bbox_from_tracker:
                    x1, y1, x2, y2 = bbox_from_tracker
                    is_near_edge = (x1 <= edge_proximity_threshold or y1 <= edge_proximity_threshold or
                                    x2 >= frame_w - edge_proximity_threshold or y2 >= frame_h - edge_proximity_threshold)

                all_detections_in_frame = self._detect_and_process_all(frame, conf_threshold)

                best_detection_in_frame = None
                highest_similarity_in_frame = -1.0

                # --- Verification using MobileCLIP2 ---
                if all_detections_in_frame and dynamic_reference_data:
                    for detection_result in all_detections_in_frame:
                        detected_crop = self._crop_object(frame, detection_result["xyxy"], detection_result["mask"])
                        if detected_crop is not None:
                            detected_embedding = self._extract_mobileclip2_embedding(detected_crop)
                            if detected_embedding is not None:
                                # Compare against all known references
                                current_avg_sim = np.mean([
                                    self._calculate_similarity(detected_embedding, ref_emb, detected_crop, ref_crop)
                                    for ref_emb, ref_crop in dynamic_reference_data
                                ])
                                if current_avg_sim > highest_similarity_in_frame:
                                    highest_similarity_in_frame = current_avg_sim
                                    best_detection_in_frame = detection_result

                detection_is_valid = False
                if best_detection_in_frame:
                    final_similarity_score = highest_similarity_in_frame
                    if final_similarity_score >= highest_similarity_score * similarity_neg_threshold_factor:
                        detection_is_valid = True
                        best_crop = self._crop_object(frame, best_detection_in_frame["xyxy"], best_detection_in_frame["mask"])

                        if best_crop is not None:
                            # --- Dynamic Model & Reference Updates ---
                            if final_similarity_score > highest_similarity_score:
                                highest_similarity_score = final_similarity_score
                                # This is a very good detection, let's use it to refine the YOLOE prompt
                                best_vpe = self._extract_vpe(frame, [best_detection_in_frame["xyxy"]])
                                if best_vpe is not None:
                                    print(f"     [Frame {frame_idx}] New best prompt! Sim: {final_similarity_score:.4f}. Updating YOLOE prompt.")
                                    self.model.set_classes([class_name], best_vpe)

                            if final_similarity_score > similarity_add_threshold:
                                best_embedding = self._extract_mobileclip2_embedding(best_crop)
                                if best_embedding is not None:
                                    print(f"     -> High similarity detection (sim: {final_similarity_score:.4f}). Adding to dynamic reference set.")
                                    dynamic_reference_data.append((best_embedding, best_crop))
                                    MAX_REFERENCE_SAMPLES = 10
                                    if len(dynamic_reference_data) > MAX_REFERENCE_SAMPLES:
                                        dynamic_reference_data.pop(0)

                if detection_is_valid:
                    detection_status = "Verified & Corrected"
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(frame, best_detection_in_frame["xywh"])
                    frames_since_detection = 0
                    bbox_for_drawing = best_detection_in_frame["xyxy"]
                else:
                    if is_near_edge:
                        detection_status = "Near Edge & Lost"
                        tracker = None
                        bbox_for_drawing = None
                    else:
                        detection_status = "Detect Fail, Keep Track"
                        bbox_for_drawing = bbox_from_tracker
                        if bbox_from_tracker: frames_since_detection += 1
            else:
                bbox_for_drawing = bbox_from_tracker
                if bbox_for_drawing is not None:
                    detection_status = "Tracking"
                    frames_since_detection += 1
                else:
                    detection_status = "Track Fail"

            if bbox_for_drawing:
                x1, y1, x2, y2 = bbox_for_drawing
                detections.append({"frame": frame_idx, "x1": x1, "y1": y1, "x2": x2, "y2": y2})

            if save_frames and output_frames_dir and bbox_for_drawing is not None and (frame_idx % save_interval == 0):
                debug_frame = frame.copy()
                x1, y1, x2, y2 = map(int, bbox_for_drawing)
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                info_text = f"Frame: {frame_idx} | Status: {detection_status} | Best Sim: {highest_similarity_score:.2f}"
                cv2.putText(debug_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imwrite(os.path.join(output_frames_dir, f"frame_{frame_idx:06d}.jpg"), debug_frame)

        cap.release()
        frames_with_detection = set(d['frame'] for d in detections)
        stats = {
            'total_frames': total_frames, 'frames_with_detection': len(frames_with_detection),
            'detection_rate': len(frames_with_detection) / total_frames if total_frames > 0 else 0
        }
        return detections, stats

    # --- Analysis and Plotting functions remain unchanged ---
    def save_analysis_report(self, video_stats: Dict, output_path: str):
        with open(output_path, 'w') as f:
            f.write("="*50 + "\nANALYSIS REPORT\n" + "="*50 + "\n\n")
            for video_id, stats in sorted(video_stats.items()):
                f.write(f"Video: {video_id}\n  - Class: {stats.get('class_name', 'N/A')}\n")
                f.write(f"  - Total Frames: {stats['total_frames']}\n  - Frames with Detections: {stats['frames_with_detection']}\n")
                f.write(f"  - Detection Rate: {stats['detection_rate']*100:.2f}%\n\n")
        print(f"Analysis report saved to: {output_path}")

    def plot_detection_stats(self, video_stats: Dict, output_path: str):
        video_ids, rates = list(video_stats.keys()), [s['detection_rate'] * 100 for s in video_stats.values()]
        plt.figure(figsize=(14, 7))
        colors = ['green' if r > 50 else 'orange' if r > 20 else 'red' for r in rates]
        bars = plt.bar(video_ids, rates, color=colors)
        plt.ylabel('Detection Rate (%)'); plt.title('Object Detection Rate per Video')
        plt.xticks(rotation=45, ha='right'); plt.ylim(0, 100); plt.grid(axis='y', linestyle='--', alpha=0.7)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.1f}%', va='bottom', ha='center')
        plt.tight_layout(); plt.savefig(output_path, dpi=150); plt.close()
        print(f"Detection statistics plot saved to: {output_path}")

    def process_dataset(
        self, dataset_path: str, output_json: str = "submission.json", temp_dir: str = "temp_augmented_images",
        yolov8_conf: float = 0.3, top_k_detections: int = 3, detection_area_weight: float = 0.5,
        detection_conf_weight: float = 0.5, num_backgrounds: int = 5, objects_per_background: int = 3,
        min_aug_scale: float = 0.02, max_aug_scale: float = 0.08, prefer_high_conf_crops: bool = True,
        yoloe_conf: float = 0.1, tracker_reinit_interval: int = 30, edge_proximity_threshold: int = 10,
        save_frames: bool = True, frames_output_dir: str = "detection_frames_corrected", save_frame_interval: int = 30,
        generate_report: bool = True, generate_plots: bool = True,
    ):
        dataset_path = Path(dataset_path)
        if save_frames: os.makedirs(frames_output_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)

        submission_data, video_stats = [], {}
        video_dirs = sorted([d for d in (dataset_path / "samples").iterdir() if d.is_dir()])
        print(f"Found {len(video_dirs)} video directories to process.")

        for video_dir in video_dirs[0:]:
            video_id = video_dir.name
            print(f"\n{'='*80}\nProcessing: {video_id}")
            self.model = YOLOE(self.yoloe_model_path)
            class_name = self.extract_class_name_from_folder(video_id)
            object_images_dir, video_file = video_dir / "object_images", video_dir / "drone_video.mp4"

            if not (object_images_dir.exists() and video_file.exists()):
                print(f"  -> WARNING: Missing files for {video_id}. Skipping."); continue
            try:
                print("  1. Extracting object crops for initial prompt...")
                object_crops = self.detect_and_extract_objects(str(object_images_dir), yolov8_conf, top_k_detections, detection_area_weight, detection_conf_weight)
                if not object_crops:
                    print("  -> WARNING: No objects detected in source images. Skipping."); continue

                print("  2. Extracting ground-truth reference data for similarity check...")
                reference_data = self.extract_reference_data(str(object_images_dir), yolov8_conf)

                print("  3. Creating augmented backgrounds for initial YOLOE prompt...")
                augmented_images, bboxes_list = self.create_augmented_backgrounds(
                    str(video_file), object_crops, num_backgrounds, objects_per_background,
                    min_aug_scale, max_aug_scale, prefer_high_conf_crops, os.path.join(temp_dir, video_id)
                )

                print("  4. Processing video with full pipeline...")
                detections, stats = self.process_video(
                    augmented_images, bboxes_list, str(video_file), class_name,
                    reference_data, yoloe_conf, tracker_reinit_interval,
                    save_frames, os.path.join(frames_output_dir, video_id) if save_frames else None, save_frame_interval,
                    edge_proximity_threshold=edge_proximity_threshold,
                    similarity_add_threshold=0.55
                )
                stats['class_name'] = class_name
                video_stats[video_id] = stats
                print(f"     -> Finished. Detection Rate: {stats['detection_rate']*100:.2f}%")
                submission_data.append({"video_id": video_id, "detections": self._group_consecutive_detections(detections)})

            except Exception as e:
                print(f"  -> ERROR processing {video_id}: {e}"); import traceback; traceback.print_exc()
                submission_data.append({"video_id": video_id, "detections": []})

        with open(output_json, 'w') as f: json.dump(submission_data, f, indent=4)
        print(f"\n{'='*80}\n✓ Processing complete! Submission saved to: {output_json}")
        if generate_report and video_stats: self.save_analysis_report(video_stats, output_json.replace('.json', '_report.txt'))
        if generate_plots and video_stats: self.plot_detection_stats(video_stats, output_json.replace('.json', '_stats.png'))
        return submission_data, video_stats


if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    pipeline = YOLOEPipeline(
        yoloe_model_path="yoloe-11l-seg.pt",
        yolov8_model_path="yolov8s.pt",
        clip_model_name="MobileCLIP2-S0",
        clip_pretrained="dfndr2b",
        seed=42
    )

    dataset_path = "/mlcv2/Datasets/ZaloAI2025/track1/public_test/"

    if not Path(dataset_path).exists():
        print("="*80 + f"\nERROR: Dataset path not found: '{dataset_path}'\n" +
              "Please update the 'dataset_path' variable in the __main__ block.\n" + "="*80)
    else:
        submission, stats = pipeline.process_dataset(
            dataset_path=dataset_path,
            output_json="submission_corrected_mobileclip_verification.json",

            # Parameters from your original code are all here
            yolov8_conf=0.1,
            top_k_detections=1,
            detection_area_weight=0.7,
            detection_conf_weight=0.3,
            num_backgrounds=1,
            objects_per_background=5,
            min_aug_scale=0.05,
            max_aug_scale=0.15,
            prefer_high_conf_crops=True,
            yoloe_conf=0.01,
            tracker_reinit_interval=5,
            edge_proximity_threshold=10,
            save_frames=False,
            save_frame_interval=5,
            generate_report=True,
            generate_plots=True
        )
