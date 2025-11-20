# ======================================================================
# PREPROCESSING STEP - preprocessing.py (MODIFIED FOR COLOR FEATURES)
# ======================================================================
import numpy as np
import cv2
import os
import re
import json
import random
from typing import List, Optional, Tuple, Dict
from pathlib import Path
import torch
import torch.nn.functional as F
from ultralytics import YOLOE, YOLO
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor
import open_clip
from PIL import Image
# MODIFIED: Changed from KMeans to MiniBatchKMeans for better performance
from sklearn.cluster import MiniBatchKMeans
import math

class DataPreprocessor:
    """Handles all preprocessing steps including reference extraction and augmentation."""
    
    def __init__(self,
                 yolov8_model_path: str = "yolov8n.pt",
                 yoloe_model_path: str = "yoloe-11s-seg.pt",
                 clip_model_name: str = "MobileCLIP2-S0",
                 clip_model_path: Optional[str] = None,
                 clip_pretrained: str = "dfndr2b",
                 seed: Optional[int] = None):
        """Initialize preprocessing models."""
        self.yolov8_model_path = yolov8_model_path
        self.yoloe_model_path = yoloe_model_path
        self.seed = seed
        if self.seed is not None:
            self._set_seed(self.seed)

        print("Initializing preprocessing models...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.detector = YOLO(self.yolov8_model_path)
        self.yoloe_model = YOLOE(self.yoloe_model_path)

        print(f"Initializing MobileCLIP2 model: {clip_model_name}...")
        
        if clip_model_path and os.path.exists(clip_model_path):
            print(f"Loading CLIP image encoder from local file: {clip_model_path}")
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                clip_model_name, pretrained=None
            )
            state_dict = torch.load(clip_model_path, map_location=self.device)
            self.clip_model.visual.load_state_dict(state_dict)
        else:
            print(f"Loading CLIP model from pretrained repository: '{clip_pretrained}'")
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                clip_model_name,
                pretrained=clip_pretrained
            )

        self.clip_model.to(self.device)
        self.clip_model.eval()
        print("Preprocessing models initialized successfully.")

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

    @staticmethod
    def _crop_object(image: np.ndarray, bbox: Tuple[float, float, float, float],
                     mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x1 >= x2 or y1 >= y2: return None
        
        crop = image[y1:y2, x1:x2]
        if mask is None: return crop
        
        full_mask = mask.squeeze() if mask.ndim == 3 else mask
        mask_crop = full_mask[y1:y2, x1:x2]
        binary_mask = (mask_crop > 0.4).astype(np.uint8)
        
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(binary_mask, kernel, iterations=2)
        alpha_channel = dilated_mask * 255
        
        if alpha_channel.shape[:2] != crop.shape[:2]:
            alpha_channel = cv2.resize(alpha_channel, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        BGRa_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
        BGRa_crop[:, :, 3] = alpha_channel
        return BGRa_crop

    def _extract_mobileclip2_embedding(self, image_crop: np.ndarray) -> Optional[torch.Tensor]:
        if image_crop is None or image_crop.size == 0: return None
        try:
            image_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGRA2RGB if image_crop.shape[2] == 4 else cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            image_tensor = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.clip_model.encode_image(image_tensor)
                embedding = F.normalize(embedding, p=2, dim=-1)
            return embedding.cpu()
        except Exception as e:
            print(f"Error during MobileCLIP2 embedding extraction: {e}")
            return None

    # ======================================================================
    # NEW METHOD: Extracts dominant color features from a single crop
    # ======================================================================
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

            # For performance, downsample if the crop is very large
            max_pixels_for_kmeans = 10000
            if pixels.shape[0] > max_pixels_for_kmeans:
                scale = np.sqrt(max_pixels_for_kmeans / pixels.shape[0])
                small_img = cv2.resize(pixels.reshape(1, -1, 3).astype(np.uint8), (0, 0), fx=scale, fy=1.0)
                pixels = small_img.reshape(-1, 3)

            # Convert to LAB color space (perceptually more uniform)
            lab_pixels = cv2.cvtColor(pixels.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB).reshape(-1, 3)
            
            # Use MiniBatchKMeans for faster clustering
            kmeans = MiniBatchKMeans(n_clusters=num_colors, random_state=self.seed, n_init='auto')
            kmeans.fit(lab_pixels)
            
            # Get cluster centers (dominant colors) and their weights
            unique_labels, counts = np.unique(kmeans.labels_, return_counts=True)
            weights = counts / counts.sum()
            dominant_colors_lab = kmeans.cluster_centers_
            
            # Sort colors by weight (most dominant first)
            sorted_indices = np.argsort(weights)[::-1]
            return dominant_colors_lab[sorted_indices], weights[sorted_indices]

        except Exception as e:
            print(f"  -> WARNING: Could not extract dominant colors. Error: {e}")
            return None

    def _extract_vpe(self, image: np.ndarray, bboxes: List[List[int]]) -> Optional[torch.Tensor]:
        self.yoloe_model.predictor = None
        if not bboxes: return None
        temp_img_path = f"temp_frame_for_vpe_{random.randint(0, 99999)}.jpg"
        cv2.imwrite(temp_img_path, image)
        try:
            visual_prompts = {'bboxes': [np.array(bboxes)], 'cls': [np.array([0] * len(bboxes))]}
            self.yoloe_model.predict(
                temp_img_path, prompts=visual_prompts, predictor=YOLOEVPSegPredictor,
                return_vpe=True, verbose=False
            )
            vpe = self.yoloe_model.predictor.vpe
        finally:
            self.yoloe_model.predictor = None
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
        return vpe

    # MODIFIED: Now extracts and returns color features as well
    def extract_reference_data(self, object_images_dir: str, 
                               conf_threshold: float = 0.1) -> List[Tuple[torch.Tensor, np.ndarray, tuple]]:
        """
        Extracts MobileCLIP2 embeddings, crops, and color features from original sample images.
        """
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
                crop = self._crop_object(img, best_box_coords)
                
                if crop is not None:
                    embedding = self._extract_mobileclip2_embedding(crop)
                    # NEW: Extract dominant color features
                    color_features = self._extract_dominant_colors(crop)
                    
                    # Only add if all features are successfully extracted
                    if embedding is not None and color_features is not None:
                        reference_data.append((embedding, crop, color_features))
                        
        print(f"     Extracted {len(reference_data)} complete reference sets (embedding, crop, color).")
        return reference_data
    def _score_detection(self, bbox: Tuple[int, int, int, int], confidence: float,
                        area_weight: float, conf_weight: float) -> float:
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        return area_weight * min(1.0, area / (640 * 640)) + conf_weight * confidence

    def detect_and_extract_objects(self, object_images_dir: str, conf_threshold: float,
                                   top_k: int, area_weight: float, 
                                   conf_weight: float) -> List[Tuple[np.ndarray, Tuple, float]]:
        """Detects objects and extracts the top-k scored CROPS for augmentation."""
        crops = []
        image_files = [f for f in os.listdir(object_images_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"     Found {len(image_files)} object images.")
        for img_file in sorted(image_files, key=self._natural_key):
            img_path = os.path.join(object_images_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            results = self.detector.predict(img_path, save=False, conf=conf_threshold, verbose=False)
            result = results[0]
            if result.boxes:
                detections = [
                    {'bbox': tuple(map(int, box)), 'conf': conf, 
                     'score': self._score_detection(tuple(map(int, box)), conf, area_weight, conf_weight)}
                    for box, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy())
                ]
                for det in sorted(detections, key=lambda x: x['score'], reverse=True)[:top_k]:
                    x1, y1, x2, y2 = det['bbox']
                    crop = img[y1:y2, x1:x2].copy()
                    if crop.size > 0:
                        crops.append((crop, det['bbox'], det['conf']))
        print(f"     Extracted {len(crops)} top-quality object crops for augmentation.")
        return crops

    def sample_video_frames(self, video_path: str, num_frames: int = 5) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        frames = []
        for _ in range(num_frames):
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)
        cap.release()
        return frames

    def jitter_brightness_contrast(self, patch: np.ndarray, jb: float = 0.35, 
                                   jc: float = 0.2) -> np.ndarray:
        b_factor = 1.0 + random.uniform(-jb, jb)
        c_factor = 1.0 + random.uniform(-jc, jc)
        patch_f = (patch.astype(np.float32) - patch.mean()) * c_factor + patch.mean()
        patch_f = np.clip(patch_f * b_factor, 0, 255).astype(np.uint8)
        return patch_f

    def create_augmented_backgrounds(self, video_path: str, object_crops: List,
                                    num_backgrounds: int, objects_per_bg: int,
                                    min_scale: float, max_scale: float,
                                    prefer_high_conf: bool,
                                    output_dir: Optional[str] = None) -> Tuple[List[np.ndarray], List[List[Tuple]]]:
        if not object_crops:
            raise ValueError("No object crops provided for augmentation.")
        bg_frames = self.sample_video_frames(video_path, num_backgrounds)
        if not bg_frames:
            raise RuntimeError("Failed to sample frames from the video.")
        augmented_images, all_bboxes = [], []
        weights = np.array([c[2] for c in object_crops])
        weights /= weights.sum() if weights.sum() > 0 else 1.0

        for bg_idx, bg in enumerate(bg_frames):
            H, W, _ = bg.shape
            bg_copy, frame_bboxes = bg.copy(), []
            for _ in range(min(objects_per_bg, len(object_crops))):
                crop_data = object_crops[np.random.choice(len(object_crops), p=weights) if prefer_high_conf 
                                        else random.randint(0, len(object_crops) - 1)]
                crop = crop_data[0]
                obj = self.jitter_brightness_contrast(crop.copy())
                scale = random.uniform(min_scale, max_scale)
                tgt_w = max(8, int(round(W * scale)))
                tgt_h = max(8, int(round(obj.shape[0] * (tgt_w / max(1, obj.shape[1])))))
                if tgt_w <= 0 or tgt_h <= 0:
                    continue
                obj_small = cv2.resize(obj, (tgt_w, tgt_h), interpolation=cv2.INTER_AREA)
                x1 = random.randint(0, max(0, W - tgt_w - 1))
                y1 = random.randint(0, max(0, H - tgt_h - 1))
                if x1 + tgt_w > W or y1 + tgt_h > H:
                    continue
                bg_copy[y1:y1+tgt_h, x1:x1+tgt_w] = obj_small
                frame_bboxes.append((x1, y1, x1+tgt_w, y1+tgt_h))
            augmented_images.append(bg_copy)
            all_bboxes.append(frame_bboxes)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                cv2.imwrite(os.path.join(output_dir, f"aug_bg_{bg_idx:02d}.jpg"), bg_copy)
        print(f"     Created {len(augmented_images)} augmented backgrounds.")
        return augmented_images, all_bboxes

    def create_initial_vpe(self, augmented_images: List[np.ndarray], 
                          bboxes_list: List[List[Tuple]]) -> torch.Tensor:
        """Creates initial VPE from augmented images."""
        initial_vpes = []
        for aug_img, bboxes in zip(augmented_images, bboxes_list):
            vpe = self._extract_vpe(aug_img, bboxes)
            if vpe is not None:
                initial_vpes.append(vpe)
        
        if not initial_vpes:
            raise RuntimeError("Could not generate an initial VPE from augmented images.")
        
        initial_vpe = torch.mean(torch.cat(initial_vpes, dim=0), dim=0, keepdim=True)
        return initial_vpe

    def extract_class_name_from_folder(self, folder_name: str) -> str:
        return re.sub(r'_\d+$', '', folder_name)

    def preprocess_video(self, video_dir: Path, output_dir: Path, config: Dict) -> Dict:
        video_id = video_dir.name
        print(f"\nPreprocessing: {video_id}")
        
        class_name = self.extract_class_name_from_folder(video_id)
        object_images_dir = video_dir / "object_images"
        video_file = video_dir / "drone_video.mp4"

        if not (object_images_dir.exists() and video_file.exists()):
            print(f"  -> WARNING: Missing files for {video_id}. Skipping.")
            return None

        try:
            print("  1. Extracting object crops for augmentation...")
            object_crops = self.detect_and_extract_objects(
                str(object_images_dir), config['yolov8_conf'], config['top_k_detections'],
                config['detection_area_weight'], config['detection_conf_weight']
            )
            if not object_crops:
                print("  -> WARNING: No objects detected for augmentation. Skipping.")
                return None

            print("  2. Extracting reference data (embeddings, crops, colors)...")
            reference_data = self.extract_reference_data(str(object_images_dir), config['yolov8_conf'])
            if not reference_data:
                print("  -> WARNING: No valid reference data could be extracted. Skipping.")
                return None

            print("  3. Creating augmented backgrounds...")
            augmented_images, bboxes_list = self.create_augmented_backgrounds(
                str(video_file), object_crops, config['num_backgrounds'], config['objects_per_background'],
                config['min_aug_scale'], config['max_aug_scale'], config['prefer_high_conf_crops']
            )

            print("  4. Creating initial VPE...")
            initial_vpe = self.create_initial_vpe(augmented_images, bboxes_list)

            # Save preprocessed data
            video_output_dir = output_dir / video_id
            video_output_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save(initial_vpe, video_output_dir / "initial_vpe.pt")
            
            # Unpack the reference data for saving
            ref_embeddings = torch.cat([emb for emb, _, _ in reference_data], dim=0)
            ref_crops = [crop for _, crop, _ in reference_data]
            ref_color_features = [colors for _, _, colors in reference_data]

            torch.save(ref_embeddings, video_output_dir / "reference_embeddings.pt")
            np.save(video_output_dir / "reference_crops.npy", np.array(ref_crops, dtype=object))
            # NEW: Save the extracted color features to a .npy file
            np.save(video_output_dir / "reference_color_features.npy", np.array(ref_color_features, dtype=object))
            
            metadata = {
                'video_id': video_id, 'class_name': class_name, 'video_path': str(video_file),
                'num_references': len(reference_data)
            }
            with open(video_output_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=4)
            
            print(f"  ✓ Preprocessing complete for {video_id}")
            return metadata

        except Exception as e:
            print(f"  -> ERROR preprocessing {video_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def preprocess_dataset(self, dataset_path: str, output_dir: str, config: Dict):
        dataset_path = Path(dataset_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        video_dirs = sorted([d for d in (dataset_path / "samples").iterdir() if d.is_dir()])
        print(f"Found {len(video_dirs)} video directories to preprocess.")

        all_metadata = [md for video_dir in video_dirs if (md := self.preprocess_video(video_dir, output_dir, config)) is not None]

        with open(output_dir / "dataset_metadata.json", 'w') as f:
            json.dump(all_metadata, f, indent=4)
        
        print(f"\n{'='*80}\n✓ Preprocessing complete! Data saved to: {output_dir}")
        return all_metadata


if __name__ == "__main__":
    clip_encoder_path = "/mlcv2/WorkingSpace/Personal/quannh/Project/Project/ZaloAI2025/RealtimeFewshotDetection_DroneAspective/yoloe/models/mobileclip2_image_encoder_fp16.pt"

    if not os.path.exists(clip_encoder_path):
        print(f"ERROR: CLIP model file not found at '{clip_encoder_path}'.")
    else:
        preprocessor = DataPreprocessor(
            yolov8_model_path="yolov8s.pt",
            yoloe_model_path="yoloe-11l-seg.pt",
            clip_model_name="MobileCLIP2-S0",
            clip_model_path=clip_encoder_path,
            clip_pretrained="dfndr2b",
            seed=42
        )

        config = {
            'yolov8_conf': 0.1, 'top_k_detections': 1, 'detection_area_weight': 0.7,
            'detection_conf_weight': 0.3, 'num_backgrounds': 1, 'objects_per_background': 5,
            'min_aug_scale': 0.05, 'max_aug_scale': 0.15, 'prefer_high_conf_crops': True,
        }

        dataset_path = "/mlcv2/Datasets/ZaloAI2025/track1/public_test/"
        output_dir = "preprocessed_data"

        preprocessor.preprocess_dataset(dataset_path, output_dir, config)