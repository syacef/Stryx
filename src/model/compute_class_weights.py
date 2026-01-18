"""
Compute class weights for handling imbalanced SA-FARI dataset.
Only considers annotations for videos that exist on disk.
"""

import json
import os
from collections import Counter
import numpy as np


def compute_class_weights(
    json_path="src/model/data/annotated/train/sa_fari_train.json",
    data_root="src/model/data/train",
    output_file="class_weights.json"
):
    """
    Compute class weights based on inverse frequency for available videos.
    
    Args:
        json_path: Path to COCO annotations JSON
        data_root: Root directory containing video folders
        output_file: Output JSON file to save weights
    
    Returns:
        dict: Dictionary with class statistics and weights
    """
    
    # Load annotations
    print(f"Loading annotations from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get available video folders on disk
    print(f"Scanning {data_root} for available videos...")
    available_folders = set(os.listdir(data_root))
    print(f"Found {len(available_folders)} video folders on disk")
    
    # Create mapping from video_id to folder name
    video_map = {}
    for video_entry in data['videos']:
        video_id = video_entry['id']
        folder_name = video_entry.get('file_name', f"sa_fari_{video_id:06d}")
        if folder_name in available_folders:
            video_map[video_id] = folder_name
    
    print(f"Mapped {len(video_map)} videos from JSON to available folders")
    
    # Count category occurrences for available videos only
    category_counts = Counter()
    total_annotations = 0
    
    for ann in data['annotations']:
        video_id = ann['video_id']
        if video_id in video_map:  # Only count if video exists
            category_id = ann['category_id']
            category_counts[category_id] += 1
            total_annotations += 1
    
    print(f"\nProcessed {total_annotations} annotations across {len(category_counts)} classes")
    
    # Create category_id to name mapping
    category_names = {cat['id']: cat['name'] for cat in data['categories']}
    
    # Compute weights (inverse frequency)
    num_classes = len(category_counts)
    class_weights = {}
    class_info = {}
    
    # Method 1: Inverse frequency normalized
    total_samples = sum(category_counts.values())
    
    for cat_id, count in sorted(category_counts.items()):
        # Inverse frequency weight
        weight = total_samples / (num_classes * count)
        
        cat_name = category_names.get(cat_id, f"unknown_{cat_id}")
        class_weights[cat_id] = weight
        class_info[cat_id] = {
            'name': cat_name,
            'count': count,
            'frequency': count / total_samples,
            'weight': weight
        }
    
    # Print statistics
    print("\n" + "="*80)
    print("CLASS STATISTICS (sorted by count)")
    print("="*80)
    print(f"{'Category ID':<12} {'Name':<25} {'Count':<8} {'Frequency':<12} {'Weight':<10}")
    print("-"*80)
    
    sorted_classes = sorted(class_info.items(), key=lambda x: x[1]['count'], reverse=True)
    for cat_id, info in sorted_classes:
        print(f"{cat_id:<12} {info['name']:<25} {info['count']:<8} "
              f"{info['frequency']:<12.4f} {info['weight']:<10.4f}")
    
    print("-"*80)
    print(f"Total classes: {num_classes}")
    print(f"Total annotations: {total_annotations}")
    print(f"Min count: {min(category_counts.values())}")
    print(f"Max count: {max(category_counts.values())}")
    print(f"Mean count: {np.mean(list(category_counts.values())):.2f}")
    print(f"Median count: {np.median(list(category_counts.values())):.2f}")
    
    # Compute imbalance ratio
    max_count = max(category_counts.values())
    min_count = min(category_counts.values())
    imbalance_ratio = max_count / min_count
    print(f"Imbalance ratio (max/min): {imbalance_ratio:.2f}x")
    
    # Save to file
    output_data = {
        'num_classes': num_classes,
        'total_annotations': total_annotations,
        'total_videos': len(video_map),
        'class_weights': {int(k): float(v) for k, v in class_weights.items()},
        'class_info': {
            int(k): {
                'name': v['name'],
                'count': v['count'],
                'frequency': float(v['frequency']),
                'weight': float(v['weight'])
            }
            for k, v in class_info.items()
        },
        'statistics': {
            'min_count': min_count,
            'max_count': max_count,
            'mean_count': float(np.mean(list(category_counts.values()))),
            'median_count': float(np.median(list(category_counts.values()))),
            'imbalance_ratio': float(imbalance_ratio)
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Class weights saved to {output_file}")
    
    # # Also create a PyTorch-ready tensor file
    # import torch
    
    # # Create ordered weight tensor (assuming category IDs are sequential)
    # sorted_cat_ids = sorted(class_weights.keys())
    # weight_tensor = torch.tensor([class_weights[cat_id] for cat_id in sorted_cat_ids], dtype=torch.float32)
    
    # torch_output = output_file.replace('.json', '.pt')
    # torch.save({
    #     'weights': weight_tensor,
    #     'category_ids': sorted_cat_ids,
    #     'num_classes': num_classes
    # }, torch_output)
    
    # print(f"✓ PyTorch weights saved to {torch_output}")
    
    return output_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute class weights for SA-FARI dataset")
    parser.add_argument(
        "--json_path",
        type=str,
        default="src/model/data/annotated/train/sa_fari_train.json",
        help="Path to annotations JSON file"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="src/model/data/train",
        help="Root directory containing video folders"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="class_weights.json",
        help="Output JSON file path"
    )
    
    args = parser.parse_args()
    
    compute_class_weights(
        json_path=args.json_path,
        data_root=args.data_root,
        output_file=args.output
    )
