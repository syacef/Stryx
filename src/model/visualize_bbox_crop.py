import json
import os
import random
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dataset.bbox_transforms import BboxAwareCrop


def load_annotations(json_path):
    """Load COCO-style annotations"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def get_random_crop(img, output_size=224):
    """Perform a standard random crop"""
    width, height = img.size
    
    if width < output_size or height < output_size:
        # Resize if smaller than target
        scale = max(output_size / width, output_size / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = img.resize((new_width, new_height), Image.BICUBIC)
        width, height = img.size
    
    # Random crop
    left = random.randint(0, max(0, width - output_size))
    top = random.randint(0, max(0, height - output_size))
    
    return img.crop((left, top, left + output_size, top + output_size))


def draw_bbox_on_image(img, bbox, color='red', width=3):
    """Draw bounding box on image"""
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    
    x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
    draw.rectangle([x, y, x + w, y + h], outline=color, width=width)
    
    return img_copy


def visualize_crops(json_path, data_root, num_samples=5, seed=42):
    """
    Visualize original image vs bbox-aware crop vs random crop
    
    Args:
        json_path: Path to COCO annotations
        data_root: Root directory containing video folders
        num_samples: Number of samples to visualize
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Load annotations
    print(f"Loading annotations from {json_path}...")
    data = load_annotations(json_path)
    
    # Create video_id to name mapping
    video_map = {v['id']: v['video_name'] for v in data['videos']}
    
    # Check which videos exist on disk
    available_videos = set()
    for video in data['videos']:
        video_path = os.path.join(data_root, video['video_name'])
        if os.path.exists(video_path):
            available_videos.add(video['id'])
    
    print(f"Found {len(available_videos)} videos available on disk")
    
    # Filter annotations with bboxes (bboxes is a list per frame)
    annotations_with_bbox = []
    for ann in data['annotations']:
        # Only consider annotations for videos that exist on disk
        if ann['video_id'] not in available_videos:
            continue
            
        if 'bboxes' in ann and ann['bboxes']:
            video_name = video_map[ann['video_id']]
            # Find frames with non-None bboxes
            for frame_idx, bbox in enumerate(ann['bboxes']):
                if bbox is not None and len(bbox) == 4:
                    # Check if frame file exists
                    frame_path = os.path.join(data_root, video_name, f"{frame_idx:05d}.jpg")
                    if os.path.exists(frame_path):
                        annotations_with_bbox.append({
                            'video_id': ann['video_id'],
                            'frame_idx': frame_idx,
                            'bbox': bbox,
                            'category_id': ann['category_id']
                        })
    
    # Shuffle to get variety
    random.shuffle(annotations_with_bbox)
    
    print(f"Found {len(annotations_with_bbox)} frame annotations with bboxes and available videos")
    
    # Sample random annotations
    samples = random.sample(annotations_with_bbox, min(num_samples, len(annotations_with_bbox)))
    
    # Create bbox-aware crop transform
    bbox_crop = BboxAwareCrop(output_size=224, bbox_prob=1.0)  # Always use bbox crop
    
    # Visualize each sample
    for idx, sample in enumerate(samples):
        video_id = sample['video_id']
        frame_idx = sample['frame_idx']
        video_name = video_map.get(video_id, f"video_{video_id}")
        
        # Construct frame path
        frame_path = os.path.join(
            data_root,
            video_name,
            f"{frame_idx:05d}.jpg"
        )
        
        if not os.path.exists(frame_path):
            print(f"⚠ Frame not found: {frame_path}")
            continue
        
        # Load image
        img = Image.open(frame_path).convert('RGB')
        
        # Get bbox (in COCO format: [x, y, width, height])
        bbox_coco = sample['bbox']
        bbox = {
            'x': bbox_coco[0],
            'y': bbox_coco[1],
            'width': bbox_coco[2],
            'height': bbox_coco[3]
        }
        
        # Get category info
        category_id = sample['category_id']
        category_name = next(
            (cat['name'] for cat in data['categories'] if cat['id'] == category_id),
            f"class_{category_id}"
        )
        
        # Calculate bbox area percentage
        img_area = img.size[0] * img.size[1]
        bbox_area = bbox['width'] * bbox['height']
        bbox_percentage = (bbox_area / img_area) * 100
        
        # Create crops
        img_with_bbox = draw_bbox_on_image(img, bbox, color='red', width=4)
        
        # BboxAware crop (with different margins)
        bbox_crop_tight = BboxAwareCrop(output_size=224, bbox_prob=1.0, margin_range=(1.2, 1.2))
        bbox_crop_medium = BboxAwareCrop(output_size=224, bbox_prob=1.0, margin_range=(1.5, 1.5))
        bbox_crop_loose = BboxAwareCrop(output_size=224, bbox_prob=1.0, margin_range=(2.0, 2.0))
        
        crop_tight = bbox_crop_tight(img.copy(), bbox)
        crop_medium = bbox_crop_medium(img.copy(), bbox)
        crop_loose = bbox_crop_loose(img.copy(), bbox)
        
        # Random crop
        random_crop = get_random_crop(img.copy(), output_size=224)
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(
            f"Sample {idx+1}: {category_name}\n"
            f"Video: {video_name}, Frame: {frame_idx}, BBox: {bbox_percentage:.2f}% of image",
            fontsize=14, fontweight='bold'
        )
        
        # Original with bbox
        axes[0, 0].imshow(img_with_bbox)
        axes[0, 0].set_title(f"Original ({img.size[0]}×{img.size[1]}px)\nBBox: {bbox['width']:.0f}×{bbox['height']:.0f}px", fontsize=10)
        axes[0, 0].axis('off')
        
        # BboxAware crops
        axes[0, 1].imshow(crop_tight)
        axes[0, 1].set_title("BboxAware Crop (margin=1.2)\nTight: 20% padding", fontsize=10)
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(crop_medium)
        axes[0, 2].set_title("BboxAware Crop (margin=1.5)\nMedium: 50% padding", fontsize=10)
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(crop_loose)
        axes[1, 0].set_title("BboxAware Crop (margin=2.0)\nLoose: 100% padding", fontsize=10)
        axes[1, 0].axis('off')
        
        # Random crop
        axes[1, 1].imshow(random_crop)
        axes[1, 1].set_title("Random Crop\n(may miss animal)", fontsize=10)
        axes[1, 1].axis('off')
        
        # Info panel
        axes[1, 2].axis('off')
        info_text = (
            f"Bbox Info:\n"
            f"  Position: ({bbox['x']:.0f}, {bbox['y']:.0f})\n"
            f"  Size: {bbox['width']:.0f}×{bbox['height']:.0f}px\n"
            f"  Area: {bbox_area:.0f}px² ({bbox_percentage:.2f}%)\n"
            f"\n"
            f"Margin Calculation:\n"
            f"  Base size: {max(bbox['width'], bbox['height']):.0f}px\n"
            f"  × 1.2 = {max(bbox['width'], bbox['height']) * 1.2:.0f}px crop\n"
            f"  × 1.5 = {max(bbox['width'], bbox['height']) * 1.5:.0f}px crop\n"
            f"  × 2.0 = {max(bbox['width'], bbox['height']) * 2.0:.0f}px crop\n"
            f"\n"
            f"Training Strategy:\n"
            f"  70% bbox-aware crop\n"
            f"  30% random crop\n"
            f"  Margin: random(1.2, 2.0)"
        )
        axes[1, 2].text(0.1, 0.5, info_text, fontsize=9, family='monospace',
                       verticalalignment='center', bbox=dict(boxstyle='round', 
                       facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f'bbox_crop_comparison_sample_{idx+1}.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved: bbox_crop_comparison_sample_{idx+1}.png")
        plt.show()
    
    print(f"\n✓ Visualized {len(samples)} samples")


if __name__ == "__main__":
    # Paths
    json_path = "src/model/data/annotated/train/sa_fari_train.json"
    data_root = "src/model/data/train"
    
    print("=" * 80)
    print("BBOX-AWARE CROP VISUALIZATION")
    print("=" * 80)
    
    # Visualize 5 random samples
    visualize_crops(
        json_path=json_path,
        data_root=data_root,
        num_samples=5,
        seed=42
    )
