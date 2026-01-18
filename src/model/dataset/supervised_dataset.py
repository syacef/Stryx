import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import functional as F

class SafariSupervisedDataset(Dataset):
    def __init__(self, json_path, frames_root, labels_txt, num_frames=8, transform=None, 
                 bbox_json_path=None):
        """
        Args:
            json_path: Path to sa_fari_train_ext.json
            frames_root: Root directory for video frames
            labels_txt: Path to labels.txt
            num_frames: Number of frames to sample per video
            transform: Callable transform (can be BboxAwareTransform)
            bbox_json_path: Path to COCO-style annotations JSON with bboxes
        """
        with open(labels_txt, "r") as f:
            self.classes = [line.strip() for line in f if line.strip()]
        
        self.species_to_idx = {name: i for i, name in enumerate(self.classes)}

        with open(json_path, "r") as f:
            metadata = json.load(f)

        self.frames_root = frames_root
        self.num_frames = num_frames
        self.transform = transform
        self.valid_samples = []
        
        # Load bbox annotations if provided
        self.bbox_data = {}
        if bbox_json_path and os.path.exists(bbox_json_path):
            print(f"Loading bbox annotations from {bbox_json_path}...")
            with open(bbox_json_path, 'r') as f:
                bbox_annotations = json.load(f)
            
            # Build video_id -> bboxes mapping
            for ann in bbox_annotations.get('annotations', []):
                video_id = ann['video_id']
                if video_id not in self.bbox_data:
                    self.bbox_data[video_id] = []
                
                # Store bbox in [x, y, width, height] format
                bbox = ann.get('bboxes', [])
                if bbox:
                    # bboxes is a list of bbox per frame
                    self.bbox_data[video_id].append(bbox)
            
            print(f"Loaded bbox data for {len(self.bbox_data)} videos")

        for item in metadata.get("video_np_pairs", []):
            if item.get("num_masklets", 0) > 0:
                video_id = item["video_id"]
                species_name = item["noun_phrase"]
                
                folder_name = f"sa_fari_{video_id:06d}"
                folder_path = os.path.join(self.frames_root, folder_name)

                if os.path.isdir(folder_path):
                    frames = sorted([
                        f for f in os.listdir(folder_path) 
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))
                    ])
                    
                    if len(frames) >= 1:
                        self.valid_samples.append({
                            "label": self.species_to_idx.get(species_name, -1),
                            "folder_path": folder_path,
                            "frame_names": frames,
                            "video_id": video_id
                        })

        self.valid_samples = [s for s in self.valid_samples if s["label"] != -1]
        print(f"Dataset initialized: {len(self.valid_samples)} valid videos found.")

    def __len__(self):
        return len(self.valid_samples)
    
    def get_bbox_for_frame(self, video_id, frame_idx):
        """Get bbox for specific frame of a video (public method for transforms)"""
        if video_id not in self.bbox_data:
            return None
        
        # bbox_data[video_id] is a list of bbox lists (one per annotation track)
        bbox_tracks = self.bbox_data[video_id]
        if not bbox_tracks:
            return None
        
        # Get first track
        bboxes = bbox_tracks[0]
        if frame_idx >= len(bboxes):
            # Use last available bbox if frame_idx exceeds
            frame_idx = len(bboxes) - 1
        
        bbox = bboxes[frame_idx]
        if bbox and len(bbox) == 4:
            return {
                'x': bbox[0],
                'y': bbox[1],
                'width': bbox[2],
                'height': bbox[3]
            }
        return None

    def __getitem__(self, idx):
        sample = self.valid_samples[idx]
        video_id = sample["video_id"]
        
        indices = torch.linspace(
            0, len(sample["frame_names"]) - 1, self.num_frames
        ).long()

        clip = []
        for i in indices:
            frame_path = os.path.join(sample["folder_path"], sample["frame_names"][i])
            img = Image.open(frame_path).convert("RGB")
            
            # Get bbox for this frame
            bbox = self.get_bbox_for_frame(video_id, int(i))
            
            # Apply transform (can be BboxAwareTransform that uses bbox)
            if not self.transform:
                img = img.resize((224, 224), Image.BILINEAR)
                clip.append(F.to_tensor(img))
            else:
                # Check if transform accepts bbox parameter
                try:
                    transformed = self.transform(img, bbox=bbox)
                except TypeError:
                    # Fallback for standard transforms without bbox support
                    transformed = self.transform(img)
                clip.append(transformed)

        clip = torch.stack(clip)
        
        return clip, sample["label"]

