import json
import argparse
import os
from collections import defaultdict

def find_small_objects(json_path, area_threshold_pixels=256, data_root='src/model/data/train'):
    """
    Finds videos containing annotations with bounding box area smaller than threshold.
    Default threshold 1024 pixels corresponds to 32x32 box (COCO 'small' definition).
    Only reports videos that actually exist on disk.
    """
    print(f"Loading annotations from {json_path}...")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {json_path}")
        return

    # Get list of actual video folders on disk
    if os.path.exists(data_root):
        available_folders = set(os.listdir(data_root))
        print(f"Found {len(available_folders)} video folders on disk in {data_root}")
    else:
        print(f"Warning: data_root {data_root} not found, will list all videos in JSON")
        available_folders = None

    # Map categories to names
    cat_map = {c['id']: c.get('noun_phrase', c.get('name', str(c['id']))) for c in data.get('categories', [])}
    
    # Map video IDs to names (filter to only those on disk)
    video_map = {}
    for v in data.get('videos', []):
        if available_folders is None or v['video_name'] in available_folders:
            video_map[v['id']] = v

    small_object_videos = defaultdict(list)
    total_small_anns = 0

    print(f"Scanning for annotations with area < {area_threshold_pixels} px^2...")

    for ann in data.get('annotations', []):
        video_id = ann['video_id']
        
        # Skip if video doesn't exist on disk
        if video_id not in video_map:
            continue
            
        category_name = cat_map.get(ann['category_id'], f"ID {ann['category_id']}")
        
        # Check standard bounding box format
        bboxes = []
        if 'bboxes' in ann: # Method 1: Track-based list of bboxes
            bboxes = [b for b in ann['bboxes'] if b]
        elif 'bbox' in ann: # Method 2: Single bbox
            bboxes = [ann['bbox']]
            
        for bbox in bboxes:
            if not bbox: continue
            
            # width * height
            area = bbox[2] * bbox[3]
            
            if area < area_threshold_pixels:
                small_object_videos[video_id].append({
                    'category': category_name,
                    'area': area,
                    'bbox': bbox,
                    'ann_id': ann['id']
                })
                total_small_anns += 1

    # Report results
    print(f"\nFound {len(small_object_videos)} videos with small objects.")
    print(f"Total small object instances (bboxes): {total_small_anns}")
    
    print("\n--- Videos with Small Objects (Top 20 by count) ---")
    
    # Sort videos by number of small annotations
    sorted_videos = sorted(small_object_videos.items(), key=lambda x: len(x[1]), reverse=True)
    
    for vid_id, small_anns in sorted_videos[:20]:
        vid_info = video_map.get(vid_id, {})
        vid_name = vid_info.get('video_name', f"ID {vid_id}")
        
        # Get unique categories found small in this video
        cats = list(set([x['category'] for x in small_anns]))
        min_area = min([x['area'] for x in small_anns])
        
        print(f"Video: {vid_name} (ID: {vid_id})")
        print(f"  - Count: {len(small_anns)} small frames")
        print(f"  - Categories: {', '.join(cats)}")
        print(f"  - Smallest Area: {min_area:.1f} px^2")
        print("-" * 30)

    # Optional: Save list to file
    output_file = "features_small_objects.txt"
    with open(output_file, "w") as f:
        for vid_id, _ in sorted_videos:
            vid_name = video_map.get(vid_id, {}).get('video_name', str(vid_id))
            f.write(f"{vid_name}\n")
    print(f"\nFull list of video names saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find videos with small annotations")
    parser.add_argument('--json_path', type=str, 
                        default='src/model/data/annotated/train/sa_fari_train.json',
                        help='Path to annotation JSON')
    parser.add_argument('--threshold', type=int, default=1024, 
                        help='Area threshold in pixels (default 1024 for 32x32)')
    parser.add_argument('--data_root', type=str, default='src/model/data/train',
                        help='Path to video folders')
    
    args = parser.parse_args()
    
    # Fallback logic if default path doesn't exist
    if not os.path.exists(args.json_path):
        alt_path = 'src/model/data/annotated/test/sa_fari_test.json'
        if os.path.exists(alt_path):
            print(f"Default train file not found, switching to: {alt_path}")
            args.json_path = alt_path
            args.data_root = 'src/model/data/test'

    find_small_objects(args.json_path, args.threshold, args.data_root)
