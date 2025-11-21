# ======================================================================
# OPTIMIZED INFERENCE STEP - inference.py
# ======================================================================
import numpy as np
import cv2
import os
import json
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
from pathlib import Path
import matplotlib.pyplot as plt
from ultralytics import YOLOE
import open_clip
from PIL import Image
import math
from sklearn.cluster import MiniBatchKMeans
import random

import time
from functools import wraps
def seed_everything(seed=42):
 random.seed(seed)
 os.environ['PYTHONHASHSEED'] = str(seed)
 np.random.seed(seed)
 torch.manual_seed(seed)
 torch.cuda.manual_seed(seed)
 torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.deterministic = True
 torch.backends.cudnn.benchmark = False
seed_everything(42) # Ví dụ cho seed bằng 42
# ======================================================================
# OPTIMIZATION: Use process pool for CPU-bound tasks
# ======================================================================
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import cpu_count

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__}: {time.time() - start:.4f}s")
        return result
    return wrapper

class VideoInference:
    """Handles video inference using preprocessed data."""
    
    def __init__(self,
                 yoloe_model_path: str = "yoloe-11s-seg.pt",
                 clip_model_name: str = "MobileCLIP2-S0",
                 clip_pretrained: str = "dfndr2b",
                 clip_encoder_path: str = "models/mobileclip2_image_encoder_fp16.pt",
                 batch_size: int = 8,
                 config: Dict = None):
        """Initialize inference models."""
        print("Initializing inference models...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.yoloe_model_path = yoloe_model_path
        self.model = YOLOE(self.yoloe_model_path)
        self.model.predictor = None
        self.batch_size = batch_size
        self.config = config if config is not None else self._get_default_config()

        print(f"Initializing MobileCLIP2 model: {clip_model_name}...")
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            clip_model_name,
            pretrained=None
        )

        print(f"Loading pre-saved image encoder weights from: {clip_encoder_path}")
        if os.path.exists(clip_encoder_path):
            encoder_state_dict = torch.load(clip_encoder_path, map_location=self.device)
            self.clip_model.visual.load_state_dict(encoder_state_dict)
            print("Successfully loaded custom image encoder weights.")
        else:
            print(f"WARNING: Encoder weights file not found at '{clip_encoder_path}'.")

        self.clip_model.to(self.device)
        self.clip_model.eval()

        # OPTIMIZATION: Enable TF32 for faster computation on Ampere+ GPUs
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.tracker = None
        self.frames_since_detection = 0
        self.highest_similarity_score = 0.2
        self.dynamic_reference_data = []
        self.frame_w = 0
        self.frame_h = 0

        # OPTIMIZATION: Pre-allocate tensor storage for reference embeddings
        self.reference_embeddings_tensor = None
        
        # OPTIMIZATION: Cache frequently used kernel
        self.dilation_kernel = np.ones((3, 3), np.uint8)
        
        # OPTIMIZATION: Use ThreadPoolExecutor (better for I/O and small tasks)
        num_workers = min(8, cpu_count())  # Limit workers to avoid overhead
        print(f"Initializing ThreadPoolExecutor with {num_workers} workers.")
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

        print("Inference models initialized successfully.")

    def __del__(self):
        """Destructor to clean up resources."""
        if hasattr(self, 'executor'):
            print("Shutting down ThreadPoolExecutor...")
            self.executor.shutdown(wait=True)

    def _get_default_config(self):
        """Provides a default config if none is given."""
        return {
            'yoloe_conf': 0.01,
            'tracker_reinit_interval': 5,
            'edge_proximity_threshold': 10,
            'similarity_neg_threshold_factor': 0.95,
            'similarity_add_threshold': 0.55,
            'max_reference_samples': 10,
        }

    # ======================================================================
# OPTIMIZED INFERENCE STEP - inference.py (Corrected)
# ======================================================================


    def initialize_for_streaming(self, preprocessed_data: Dict, frame_width: int, frame_height: int):
        """
        Initializes the model and state for processing a new video stream.
        """
        print("Initializing state for new video stream...")
        
        initial_vpe = preprocessed_data['initial_vpe']
        class_name = preprocessed_data['class_name']
        reference_embeddings = preprocessed_data['reference_embeddings']
        reference_crops = preprocessed_data['reference_crops']
        reference_color_features = preprocessed_data['reference_color_features']

        # CORRECTED CODE
        # Store all three components together
        self.dynamic_reference_data = [
            (reference_embeddings[i:i+1], reference_crops[i], reference_color_features[i])
            for i in range(len(reference_crops))
        ]

        # OPTIMIZATION: Pre-stack reference embeddings into single tensor
        self.reference_embeddings_tensor = torch.cat(
            [emb for emb, _, _ in self.dynamic_reference_data], dim=0
        ).to(self.device)

        print(f"     Setting class name: '{class_name}' with initial VPE.")
        self.model.set_classes([class_name], initial_vpe)
        self.model.predictor = None

        self.tracker = None
        self.frames_since_detection = 0
        self.highest_similarity_score = 0.2
        self.frame_w = frame_width
        self.frame_h = frame_height
        print("Initialization for streaming complete.")


    # ======================================================================
    # OPTIMIZATION: Vectorized similarity computation
    # ======================================================================
    def _compute_similarities_batch(self, detected_embedding: torch.Tensor) -> float:
        """Compute similarities against all references in one batch operation."""
        if self.reference_embeddings_tensor is None or len(self.dynamic_reference_data) == 0:
            return 0.0
        
        # Move to GPU if available
        detected_embedding = detected_embedding.to(self.device)
        
        # Batch cosine similarity computation
        embedding_sims = F.cosine_similarity(
            detected_embedding.expand(self.reference_embeddings_tensor.shape[0], -1),
            self.reference_embeddings_tensor,
            dim=1
        )
        return embedding_sims.mean().item()

    # ======================================================================
    # OPTIMIZATION: Simplified single detection processing
    # ======================================================================
    def _process_single_detection_fast(self, detection_result: Dict, frame: np.ndarray) -> Optional[Tuple[float, Dict]]:
        """
        Fast processing of one detection with minimal overhead.
        """
        detected_crop = self._crop_object_fast(frame, detection_result["xyxy"], detection_result["mask"])
        if detected_crop is None:
            return None

        detected_embedding = self._extract_mobileclip2_embedding_fast(detected_crop)
        if detected_embedding is None:
            return None

        # OPTIMIZATION: Use vectorized batch similarity
        current_avg_sim = self._compute_similarities_batch(detected_embedding)
        return current_avg_sim, detection_result

    @timing_decorator
    def predict_streaming(self, frame_rgb_np: np.ndarray, frame_idx: int) -> Optional[List[int]]:
        """
        OPTIMIZED: Processes a single frame in a streaming context.
        """
        frame = frame_rgb_np
        bbox_for_drawing = None
        
        # OPTIMIZATION: Fast tracker update
        bbox_from_tracker = None
        if self.tracker is not None:
            success, box = self.tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in box]
                x2, y2 = x + w, y + h
                # Fast boundary check
                if x2 > 0 and y2 > 0 and x < self.frame_w and y < self.frame_h:
                    bbox_from_tracker = (
                        max(0, x),
                        max(0, y),
                        min(self.frame_w, x2),
                        min(self.frame_h, y2)
                    )
            else:
                self.tracker = None

        run_detector = (self.tracker is None) or (self.frames_since_detection >= self.config['tracker_reinit_interval'])

        if run_detector:
            # OPTIMIZATION: Fast edge proximity check
            is_near_edge = False
            if bbox_from_tracker:
                x1, y1, x2, y2 = bbox_from_tracker
                threshold = self.config['edge_proximity_threshold']
                is_near_edge = (
                    x1 <= threshold or y1 <= threshold or
                    x2 >= self.frame_w - threshold or y2 >= self.frame_h - threshold
                )

            # OPTIMIZATION: Early exit if no reference data
            if not self.dynamic_reference_data:
                return bbox_from_tracker if bbox_from_tracker else None

            all_detections_in_frame = self._detect_and_process_all(frame, self.config['yoloe_conf'])
            best_detection_in_frame = None
            highest_similarity_in_frame = -1.0

            # OPTIMIZATION: Process detections sequentially (faster for small batches)
            if all_detections_in_frame:
                for detection in all_detections_in_frame:
                    result = self._process_single_detection_fast(detection, frame)
                    if result is not None:
                        similarity, det = result
                        if similarity > highest_similarity_in_frame:
                            highest_similarity_in_frame = similarity
                            best_detection_in_frame = det

            detection_is_valid = False
            if best_detection_in_frame:
                final_similarity_score = highest_similarity_in_frame
                threshold = self.highest_similarity_score * self.config['similarity_neg_threshold_factor']
                
                if final_similarity_score >= threshold:
                    detection_is_valid = True
                    
                    # OPTIMIZATION: Only crop if needed for reference update
                    if final_similarity_score > self.config['similarity_add_threshold']:
                        best_crop = self._crop_object_fast(frame, best_detection_in_frame["xyxy"], best_detection_in_frame["mask"])
                        
                        if best_crop is not None:
                            if final_similarity_score > self.highest_similarity_score:
                                self.highest_similarity_score = final_similarity_score
                            
                            # NEW CODE (CORRECTED)

                            best_embedding = self._extract_mobileclip2_embedding_fast(best_crop)
                            # NEW: Extract color features for the new reference sample
                            new_color_features = self._extract_dominant_colors(best_crop)
                            
                            # Only add if ALL required data is present
                            if best_embedding is not None and new_color_features is not None:
                                # Append the COMPLETE 3-element tuple to maintain consistency
                                self.dynamic_reference_data.append((best_embedding, best_crop, new_color_features))
                                
                                # OPTIMIZATION: Update stacked tensor
                                self.reference_embeddings_tensor = torch.cat([
                                    self.reference_embeddings_tensor,
                                    best_embedding.to(self.device)
                                ], dim=0)
                                
                                # Manage the size of the dynamic reference set
                                if len(self.dynamic_reference_data) > self.config['max_reference_samples']:
                                    self.dynamic_reference_data.pop(0)
                                    self.reference_embeddings_tensor = self.reference_embeddings_tensor[1:]

            if detection_is_valid:
                self.tracker = cv2.TrackerKCF_create()
                self.tracker.init(frame, best_detection_in_frame["xywh"])
                self.frames_since_detection = 0
                bbox_for_drawing = best_detection_in_frame["xyxy"]
            else:
                if is_near_edge:
                    self.tracker = None
                    bbox_for_drawing = None
                else:
                    bbox_for_drawing = bbox_from_tracker
                    if bbox_for_drawing:
                        self.frames_since_detection += 1
        else:
            bbox_for_drawing = bbox_from_tracker
            if bbox_for_drawing is not None:
                self.frames_since_detection += 1

        if bbox_for_drawing:
            return [int(c) for c in bbox_for_drawing]
        else:
            return None

    # ======================================================================
    # OPTIMIZATION: Fast crop function with reduced checks
    # ======================================================================
    def _crop_object_fast(self, image: np.ndarray, bbox: Tuple[float, float, float, float],
                     mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        # Fast boundary clipping
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        if x1 >= x2 or y1 >= y2:
            return None
        
        crop = image[y1:y2, x1:x2]
        
        if mask is None:
            return crop
        
        # Fast mask processing
        full_mask = mask.squeeze() if mask.ndim == 3 else mask
        mask_crop = full_mask[y1:y2, x1:x2]
        
        # Fast threshold
        binary_mask = (mask_crop > 0.4).astype(np.uint8) if mask_crop.dtype != np.uint8 else (mask_crop > 0).astype(np.uint8)
        
        # Use cached kernel
        dilated_mask = cv2.dilate(binary_mask, self.dilation_kernel, iterations=2)
        alpha_channel = dilated_mask * 255
        
        if alpha_channel.shape[:2] != crop.shape[:2]:
            alpha_channel = cv2.resize(alpha_channel, (crop.shape[1], crop.shape[0]), 
                                      interpolation=cv2.INTER_NEAREST)
        
        BGRa_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
        BGRa_crop[:, :, 3] = alpha_channel
        return BGRa_crop

    # ======================================================================
    # OPTIMIZATION: Fast embedding extraction with torch.inference_mode
    # ======================================================================
    def _extract_mobileclip2_embedding_fast(self, image_crop: np.ndarray) -> Optional[torch.Tensor]:
        if image_crop is None or image_crop.size == 0:
            return None
        try:
            # Fast color conversion
            if image_crop.shape[2] == 4:
                image_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGRA2RGB)
            else:
                image_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
            
            pil_image = Image.fromarray(image_crop)
            image_tensor = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)

            # OPTIMIZATION: Use inference_mode for faster inference
            with torch.inference_mode():
                embedding = self.clip_model.encode_image(image_tensor)
                embedding = F.normalize(embedding, p=2, dim=-1)
            
            return embedding.float()
            
        except Exception as e:
            print(f"Error during MobileCLIP2 embedding extraction: {e}")
            return None

    @staticmethod
    def _clip_bbox_to_frame(bbox: Tuple[int, int, int, int], 
                           frame_width: int, frame_height: int) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        return max(0, x1), max(0, y1), min(frame_width, x2), min(frame_height, y2)

    @staticmethod
    def _crop_object(image: np.ndarray, bbox: Tuple[float, float, float, float],
                     mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(int(round(x1)), w - 1))
        y1 = max(0, min(int(round(y1)), h - 1))
        x2 = max(0, min(int(round(x2)), w))
        y2 = max(0, min(int(round(y2)), h))
        if x1 >= x2 or y1 >= y2:
            return None
        crop = image[y1:y2, x1:x2]
        if mask is None:
            return crop
        full_mask = mask
        if full_mask.ndim == 3:
            full_mask = full_mask.squeeze()
        mask_crop = full_mask[y1:y2, x1:x2]
        if mask_crop.dtype != np.uint8:
            binary_mask = (mask_crop > 0.4).astype(np.uint8)
        else:
            binary_mask = (mask_crop > 0).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(binary_mask, kernel, iterations=2)
        alpha_channel = dilated_mask * 255
        if alpha_channel.shape[:2] != crop.shape[:2]:
            alpha_channel = cv2.resize(alpha_channel, (crop.shape[1], crop.shape[0]), 
                                      interpolation=cv2.INTER_NEAREST)
        BGRa_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
        BGRa_crop[:, :, 3] = alpha_channel
        return BGRa_crop

    def _extract_mobileclip2_embedding(self, image_crop: np.ndarray) -> Optional[torch.Tensor]:
        if image_crop is None or image_crop.size == 0:
            return None
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
            return embedding.float().cpu() 
            
        except Exception as e:
            print(f"Error during MobileCLIP2 embedding extraction: {e}")
            return None
    def _extract_dominant_colors(self, crop: np.ndarray, num_colors: int = 5) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Extracts dominant colors from a crop using MiniBatchKMeans in LAB color space.
        
        Args:
            crop (np.ndarray): The input image crop (can be BGR or BGRA).
            num_colors (int): The number of dominant colors to extract.
            
        Returns:
            A tuple containing (dominant_colors_lab, weights) or None if extraction fails.
        """
        if crop is None or crop.size == 0:
            return None
        try:
            # Use the alpha channel as a mask if it exists
            if crop.shape[2] == 4:
                mask = crop[:, :, 3] > 0
                if not np.any(mask): return None
                pixels = crop[:, :, :3][mask]
            else:
                pixels = crop.reshape(-1, 3)

            # Ensure there are enough pixels to cluster
            if pixels.shape[0] < num_colors:
                return None
            
            # Convert to LAB color space
            lab_pixels = cv2.cvtColor(pixels.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB).reshape(-1, 3)
            
            # Use MiniBatchKMeans for faster clustering
            kmeans = MiniBatchKMeans(n_clusters=num_colors, random_state=42, n_init='auto')
            kmeans.fit(lab_pixels)
            
            # Get cluster centers (dominant colors) and their weights
            unique_labels, counts = np.unique(kmeans.labels_, return_counts=True)
            weights = counts / counts.sum()
            dominant_colors_lab = kmeans.cluster_centers_
            
            # Sort colors by weight (most dominant first)
            sorted_indices = np.argsort(weights)[::-1]
            return dominant_colors_lab[sorted_indices], weights[sorted_indices]

        except Exception as e:
            print(f"  -> WARNING: Could not extract dominant colors during inference. Error: {e}")
            return None
    def _calculate_color_similarity(self, crop1: np.ndarray, crop2: np.ndarray, 
                                   num_colors: int = 2, seed: int = 42) -> float:
        """Calculate color similarity using CIEDE2000."""
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
            def get_dominant_colors(crop: np.ndarray):
                if crop.shape[2] == 4:
                    mask = crop[:, :, 3] > 0
                    if not np.any(mask): return None, None
                    bgr_pixels = crop[:, :, :3][mask]
                else:
                    bgr_pixels = crop.reshape(-1, 3)

                if bgr_pixels.shape[0] < num_colors: return None, None
                max_pixels = 5000
                if bgr_pixels.shape[0] > max_pixels:
                    temp_img = bgr_pixels.reshape(1, -1, 3).astype(np.uint8)
                    scale = np.sqrt(max_pixels / bgr_pixels.shape[0])
                    small_img = cv2.resize(temp_img, (0, 0), fx=scale, fy=1.0, interpolation=cv2.INTER_AREA)
                    bgr_pixels = small_img.reshape(-1, 3)

                lab_pixels = cv2.cvtColor(bgr_pixels.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB).reshape(-1, 3)
                kmeans = MiniBatchKMeans(n_clusters=num_colors, random_state=seed, n_init='1')
                kmeans.fit(lab_pixels)
                unique_labels, counts = np.unique(kmeans.labels_, return_counts=True)
                weights = counts / counts.sum()
                dominant_colors_lab = kmeans.cluster_centers_
                sorted_indices = np.argsort(weights)[::-1]
                return dominant_colors_lab[sorted_indices], weights[sorted_indices]

            dom_colors1, weights1 = get_dominant_colors(crop1)
            dom_colors2, weights2 = get_dominant_colors(crop2)

            if dom_colors1 is None or dom_colors2 is None: return 0.0

            def calculate_palette_distance(colors_from, weights_from, colors_to):
                total_distance = 0.0
                for i in range(len(colors_from)):
                    color_from_lab_cv, weight_from = colors_from[i], weights_from[i]
                    L1, a1, b1 = color_from_lab_cv[0] * (100.0/255.0), color_from_lab_cv[1]-128.0, color_from_lab_cv[2]-128.0
                    min_dist = float('inf')
                    for color_to_lab_cv in colors_to:
                        L2, a2, b2 = color_to_lab_cv[0]*(100.0/255.0), color_to_lab_cv[1]-128.0, color_to_lab_cv[2]-128.0
                        min_dist = min(min_dist, deltaE_ciede2000(L1, a1, b1, L2, a2, b2))
                    total_distance += weight_from * min_dist
                return total_distance
            
            def calculate_coverage(colors_from, weights_from, colors_to, threshold=20.0):
                covered_weight = 0.0
                for i in range(len(colors_from)):
                    color_from_lab_cv, weight_from = colors_from[i], weights_from[i]
                    L1, a1, b1 = color_from_lab_cv[0]*(100.0/255.0), color_from_lab_cv[1]-128.0, color_from_lab_cv[2]-128.0
                    min_dist = float('inf')
                    for color_to_lab_cv in colors_to:
                        L2, a2, b2 = color_to_lab_cv[0]*(100.0/255.0), color_to_lab_cv[1]-128.0, color_to_lab_cv[2]-128.0
                        min_dist = min(min_dist, deltaE_ciede2000(L1, a1, b1, L2, a2, b2))
                    if min_dist < threshold: covered_weight += weight_from
                return covered_weight

            dist_1to2, dist_2to1 = calculate_palette_distance(dom_colors1, weights1, dom_colors2), calculate_palette_distance(dom_colors2, weights2, dom_colors1)
            total_distance = (dist_1to2 + dist_2to1) / 2.0
            coverage1, coverage2 = calculate_coverage(dom_colors1, weights1, dom_colors2), calculate_coverage(dom_colors2, weights2, dom_colors1)
            coverage_penalty = 1.0 - min(coverage1, coverage2)
            sigma = 15.0
            similarity = math.exp(-(total_distance ** 2) / (2.0 * sigma * sigma))
            similarity *= (1.0 - coverage_penalty * 0.5)
            return float(np.clip(similarity, 0.0, 1.0))

        except (cv2.error, IndexError, ValueError) as e:
            return 0.0

    def _calculate_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor,
                            crop1: np.ndarray, crop2: np.ndarray,
                            embedding_weight: float = 0.7, 
                            color_weight: float = 0.3) -> float:
        embedding_similarity = F.cosine_similarity(emb1, emb2).item()
        color_similarity = self._calculate_color_similarity(crop1, crop2)
        return (embedding_weight * embedding_similarity + color_weight * color_similarity)
        
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
    
    def process_video(self, video_path: str, preprocessed_data: Dict, 
                     config_override: Dict) -> Tuple[List[Dict], Dict]:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        self.initialize_for_streaming(preprocessed_data, frame_w, frame_h)
        
        detections = []
        save_frames = config_override.get('save_frames', False)
        output_frames_dir = config_override.get('frames_output_dir', None)
        save_interval = config_override.get('save_frame_interval', 30)
        
        if save_frames and output_frames_dir:
            os.makedirs(output_frames_dir, exist_ok=True)

        print("     Starting video processing (batch mode)...")
        for frame_idx in range(total_frames):
            ok, frame = cap.read()
            if not ok:
                break
            
            bbox = self.predict_streaming(frame, frame_idx)
            
            if bbox:
                x1, y1, x2, y2 = bbox
                detections.append({"frame": frame_idx, "x1": x1, "y1": y1, "x2": x2, "y2": y2})

                if (save_frames and output_frames_dir and (frame_idx % save_interval == 0)):
                    debug_frame = frame.copy()
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    info_text = f"Frame: {frame_idx} | Streaming Mode"
                    cv2.putText(debug_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imwrite(os.path.join(output_frames_dir, f"frame_{frame_idx:06d}.jpg"), debug_frame)

        cap.release()
        
        frames_with_detection = set(d['frame'] for d in detections)
        stats = {
            'total_frames': total_frames,
            'frames_with_detection': len(frames_with_detection),
            'detection_rate': len(frames_with_detection) / total_frames if total_frames > 0 else 0,
            'class_name': preprocessed_data['class_name']
        }
        
        return detections, stats

    @staticmethod
    def _group_consecutive_detections(detections: List[Dict]) -> List[Dict]:
        if not detections: return []
        detections.sort(key=lambda d: d["frame"])
        grouped_detections, current_group = [], [detections[0]]
        for i in range(1, len(detections)):
            if detections[i]["frame"] == detections[i-1]["frame"] + 1:
                current_group.append(detections[i])
            else:
                grouped_detections.append({"bboxes": current_group})
                current_group = [detections[i]]
        if current_group:
            grouped_detections.append({"bboxes": current_group})
        return grouped_detections

    def run_inference(self, preprocessed_dir: str, output_json: str):
        preprocessed_dir = Path(preprocessed_dir)
        with open(preprocessed_dir / "dataset_metadata.json", 'r') as f:
            dataset_metadata = json.load(f)

        submission_data, video_stats = [], {}
        print(f"Found {len(dataset_metadata)} videos to process.")

        for metadata in dataset_metadata:
            video_id = metadata['video_id']
            print(f"\n{'='*80}\nProcessing: {video_id}")
            try:
                video_dir = preprocessed_dir / video_id
                preprocessed_data = {
                    'initial_vpe': torch.load(video_dir / "initial_vpe.pt"),
                    'class_name': metadata['class_name'],
                    'reference_embeddings': torch.load(video_dir / "reference_embeddings.pt"),
                    'reference_crops': np.load(video_dir / "reference_crops.npy", allow_pickle=True),
                    'reference_color_features': np.load(video_dir / "reference_color_features.npy", allow_pickle=True)

                }
                detections, stats = self.process_video(metadata['video_path'], preprocessed_data, self.config)
                video_stats[video_id] = stats
                print(f"     -> Detection Rate: {stats['detection_rate']*100:.2f}%")
                submission_data.append({"video_id": video_id, "detections": self._group_consecutive_detections(detections)})
            except Exception as e:
                print(f"  -> ERROR processing {video_id}: {e}")
                submission_data.append({"video_id": video_id, "detections": []})
        
        output_path = Path(output_json)
        output_dir = output_path.parent
        
        print(f"Ensuring output directory exists: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(submission_data, f, indent=4)
        
        print(f"\n{'='*80}\n✓ Inference complete! Submission saved to: {output_path}")
        
        if self.config.get('generate_report', True):
            report_path = output_dir / f"{output_path.stem}_report.txt"
            self.save_analysis_report(video_stats, str(report_path))
        if self.config.get('generate_plots', True):
            plot_path = output_dir / f"{output_path.stem}_stats.png"
            self.plot_detection_stats(video_stats, str(plot_path))
        
        return submission_data, video_stats


    def save_analysis_report(self, video_stats: Dict, output_path: str):
        with open(output_path, 'w') as f:
            f.write("="*50 + "\nANALYSIS REPORT\n" + "="*50 + "\n\n")
            for video_id, stats in sorted(video_stats.items()):
                f.write(f"Video: {video_id}\n  - Class: {stats.get('class_name', 'N/A')}\n")
                f.write(f"  - Total Frames: {stats['total_frames']}\n")
                f.write(f"  - Frames with Detections: {stats['frames_with_detection']}\n")
                f.write(f"  - Detection Rate: {stats['detection_rate']*100:.2f}%\n\n")
        print(f"Analysis report saved to: {output_path}")

    def plot_detection_stats(self, video_stats: Dict, output_path: str):
        video_ids, rates = list(video_stats.keys()), [s['detection_rate']*100 for s in video_stats.values()]
        plt.figure(figsize=(14, 7))
        colors = ['green' if r > 50 else 'orange' if r > 20 else 'red' for r in rates]
        bars = plt.bar(video_ids, rates, color=colors)
        plt.ylabel('Detection Rate (%)'); plt.title('Object Detection Rate per Video')
        plt.xticks(rotation=45, ha='right'); plt.ylim(0, 100)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.1f}%', va='bottom', ha='center')
        plt.tight_layout(); plt.savefig(output_path, dpi=150); plt.close()
        print(f"Detection statistics plot saved to: {output_path}")


if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    main_config = {
        'yoloe_conf': 0.01, 'tracker_reinit_interval': 10, 'edge_proximity_threshold': 10,
        'similarity_neg_threshold_factor': 0.95, 'similarity_add_threshold': 0.9,
        'max_reference_samples': 10, 'save_frames': False, 'frames_output_dir': 'detection_frames_corrected',
        'save_frame_interval': 5, 'generate_report': True, 'generate_plots': True
    }

    inference = VideoInference(
        yoloe_model_path="models/yoloe-11l-seg.pt",
        clip_model_name="MobileCLIP2-S0",
        clip_encoder_path="models/mobileclip2_image_encoder_fp16.pt",
        config=main_config
    )

    preprocessed_dir = "preprocessed_data"
    output_json_path = "/result/submission.json"

    print("\n" + "="*80 + "\nRUNNING BATCH INFERENCE\n" + "="*80)
    if not Path(preprocessed_dir).exists():
        print(f"ERROR: Preprocessed data not found: '{preprocessed_dir}'. Please run preprocessing.py first.")
    else:
        submission, stats = inference.run_inference(
            preprocessed_dir=preprocessed_dir,
            output_json=output_json_path
        )