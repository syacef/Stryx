import json
import numpy as np
import os
from collections import Counter

def perform_eda(json_path, data_root='src/model/data/train'):
    print(f"Loading {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get list of actual video folders on disk
    if os.path.exists(data_root):
        available_folders = set(os.listdir(data_root))
        print(f"Found {len(available_folders)} video folders on disk in {data_root}")
    else:
        print(f"Warning: data_root {data_root} not found, analyzing all videos in JSON")
        available_folders = None
    
    # Filter to only videos that exist on disk
    if available_folders is not None:
        original_videos = data['videos']
        data['videos'] = [v for v in original_videos if v['video_name'] in available_folders]
        
        # Also filter annotations to only include those for available videos
        available_video_ids = {v['id'] for v in data['videos']}
        original_anns = data['annotations']
        data['annotations'] = [ann for ann in original_anns if ann['video_id'] in available_video_ids]
        
        print(f"Filtered to {len(data['videos'])} available videos (out of {len(original_videos)} in JSON)")
        print(f"Filtered to {len(data['annotations'])} annotations (out of {len(original_anns)} in JSON)")
    
    # 1. Basic Counts
    num_videos = len(data['videos'])
    num_anns = len(data['annotations'])
    num_cats = len(data['categories'])
    
    print(f"\n--- Basic Statistics (Available Videos Only) ---")
    print(f"Total Videos: {num_videos}")
    print(f"Total Annotations: {num_anns}")
    print(f"Total Categories: {num_cats}")

    # 2. Class Distribution
    cat_id_to_name = {c['id']: c.get('name', c.get('noun_phrase', str(c['id']))) for c in data['categories']}
    
    # Some annotations might use 'category_id' mapping to these classes
    cat_counts = Counter([ann['category_id'] for ann in data['annotations']])
    
    print(f"\n--- Top 10 Classes ---")
    sorted_cats = cat_counts.most_common(10)
    for cat_id, count in sorted_cats:
        name = cat_id_to_name.get(cat_id, f"ID {cat_id}")
        # Try to find name in annotations if not in categories
        if name.startswith("ID"):
             example_ann = next((a for a in data['annotations'] if a['category_id'] == cat_id), None)
             if example_ann:
                 name = example_ann.get('noun_phrase', name)
        
        print(f"{name}: {count} annotations")

    # 3. Video Length Distribution
    lengths = [v.get('length', 0) for v in data['videos']]
    print(f"\n--- Video Statistics ---")
    print(f"Average Length: {np.mean(lengths):.2f} frames")
    print(f"Min Length: {np.min(lengths)}")
    print(f"Max Length: {np.max(lengths)}")
    
    # 4. Bounding Box Analysis
    areas = []
    ratios = []
    
    image_dims = {v['id']: (v.get('width', 1920), v.get('height', 1080)) for v in data['videos']}

    for ann in data['annotations']:
        # Data format check: 'bboxes' is usually a list of boxes for tracks in this dataset
        # In the visualize_data.py, we saw 'bboxes' is a list of [x,y,w,h] or None
        if 'bboxes' in ann:
            for bbox in ann['bboxes']:
                if bbox:
                    w, h = bbox[2], bbox[3]
                    if w > 0 and h > 0:
                        areas.append(w * h)
                        ratios.append(w / h)
        # Standard COCO style might have 'bbox'
        elif 'bbox' in ann:
             w, h = ann['bbox'][2], ann['bbox'][3]
             areas.append(w * h)
             ratios.append(w / h)

    if areas:
        areas = np.array(areas)
        print(f"\n--- Bounding Box Statistics ---")
        print(f"Average Area: {np.mean(areas):.2f} pixels^2")
        print(f"Smallest Area: {np.min(areas)}")
        print(f"Largest Area: {np.max(areas)}")
        
        # Relative area (assuming 1080p usually)
        # This is rough as videos might have different sizes
        avg_img_area = 1920 * 1080 
        print(f"Avg Relative Area: {(np.mean(areas)/avg_img_area)*100:.2f}% of 1080p frame")

    # 5. Missing Annotations?
    # Check how many video frames have no annotations at all
    # This is tricky without expanding everything, but we can check if tracks cover the whole video
    print(f"\n--- Sparsity Analysis ---")
    print("Checking tracking consistency...")
    # Map video_id to number of frames
    vid_lengths = {v['id']: v.get('length', 0) for v in data['videos']}
    
    # Map video_id to set of annotated frame indices
    vid_annotated_frames = {v['id']: set() for v in data['videos']}
    
    for ann in data['annotations']:
        vid_id = ann['video_id']
        if 'bboxes' in ann:
            for i, bbox in enumerate(ann['bboxes']):
                if bbox:
                    vid_annotated_frames[vid_id].add(i)
    
    # Calculate fill rate
    fill_rates = []
    for vid_id, frames_set in vid_annotated_frames.items():
        total = vid_lengths.get(vid_id, 0)
        if total > 0:
            fill_rates.append(len(frames_set) / total)
            
    print(f"Average percentage of frames with at least one object: {np.mean(fill_rates)*100:.2f}%")

if __name__ == "__main__":
    train_path = 'src/model/data/annotated/train/sa_fari_train.json'
    data_root = 'src/model/data/train'
    
    if os.path.exists(train_path):
        perform_eda(train_path, data_root)
    else:
        print("Train file not found, trying Test file...")
        test_path = 'src/model/data/annotated/test/sa_fari_test.json'
        test_data_root = 'src/model/data/test'
        perform_eda(test_path, test_data_root)
