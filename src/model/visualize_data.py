import argparse
import json
import os
import cv2
import numpy as np
from tqdm import tqdm

def visualize_video(json_path, data_root, video_id_target, output_path):
    # Load JSON
    print(f"Loading annotations from {json_path}...")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}")
        return

    # Find video
    video = None
    if 'videos' not in data:
         print("Error: 'videos' key not found in JSON.")
         return

    for v in data['videos']:
        if v['id'] == video_id_target:
            video = v
            break
    
    if video is None:
        print(f"Video ID {video_id_target} not found in dataset. Available IDs (first 5): {[v['id'] for v in data['videos'][:5]]}...")
        return

    print(f"Processing video: {video['video_name']} (ID: {video['id']})")
    print(f"Length: {video.get('length', 'Unknown')} frames")
    
    # Get annotations for this video
    vid_anns = [ann for ann in data.get('annotations', []) if ann['video_id'] == video['id']]
    print(f"Found {len(vid_anns)} annotation tracks.")
    for ann in vid_anns:
        print(f"- {ann.get('noun_phrase', 'Unknown')} (Category {ann['category_id']})")

    # Check first frame
    if not video.get('file_names'):
        print("No file_names in video object.")
        return

    first_frame_path = os.path.join(data_root, video['file_names'][0])
    print(f"Looking for images at: {first_frame_path}")
    
    if not os.path.exists(first_frame_path):
        print(f"Error: Could not read first frame: {first_frame_path}")
        print("Please check --data_root path.")
        
        # Try to guess path if common ml structure
        alt_path = os.path.join("src", "model", "data", "test", video['file_names'][0])
        if os.path.exists(alt_path):
            print(f"Found at alternative path: {alt_path}")
            data_root = os.path.join("src", "model", "data", "test")
            first_frame_path = alt_path

    frame0 = cv2.imread(first_frame_path)
    if frame0 is None:
        print(f"Error: cv2 could not open image {first_frame_path}")
        return

    h, w, c = frame0.shape
    # Use mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 10.0, (w, h))

    # Iterate frames
    for i, rel_path in enumerate(tqdm(video['file_names'])):
        img_path = os.path.join(data_root, rel_path)
        frame = cv2.imread(img_path)
        
        if frame is None:
            # Create black frame if missing but keep going
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(frame, "Missing Frame", (50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        
        # Draw Annotations
        for ann in vid_anns:
            # Check if 'bboxes' list exists and has entry for this frame index
            if 'bboxes' in ann and i < len(ann['bboxes']):
                bbox = ann['bboxes'][i]
                if bbox: # bbox can be None or empty list
                    try:
                        bx, by, bw, bh = map(int, bbox)
                        
                        # Generate consistent color for this track/category
                        color_seed = ann['id'] 
                        np.random.seed(color_seed)
                        color = (int(np.random.randint(0, 255)), int(np.random.randint(0, 255)), int(np.random.randint(0, 255)))
                        
                        # Draw Box
                        cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), color, 3)
                        
                        # Draw Background for text
                        label = ann.get('noun_phrase', str(ann['category_id']))
                        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                        cv2.rectangle(frame, (bx, by - text_h - 10), (bx + text_w, by), color, -1)
                        
                        # Draw Text
                        cv2.putText(frame, label, (bx, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    except Exception as e:
                        pass # Ignore bad box data

        # Draw Global Frame Info
        cv2.putText(frame, f"Video ID: {video['id']}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {i}/{len(video['file_names'])}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        out.write(frame)

    out.release()
    print(f"Success! Saved visualization to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_id', type=int, default=0, help='Video ID to visualize')
    parser.add_argument('--json_path', type=str, default='src/model/data/annotated/train/sa_fari_train.json')
    parser.add_argument('--data_root', type=str, default='src/model/data/train')
    parser.add_argument('--output', type=str, default='src/model/visualization/visualization_output.mp4')
    
    args = parser.parse_args()
    visualize_video(args.json_path, args.data_root, args.video_id, args.output)
