import json
import os

# Load JSON
with open('src/model/data/annotated/train/sa_fari_train.json', 'r') as f:
    data = json.load(f)

# Get list of actual folders
actual_folders = set(os.listdir('src/model/data/train'))

# Get videos from JSON that exist locally
existing_videos = [v for v in data['videos'] if v['video_name'] in actual_folders]

print(f"Total videos in JSON: {len(data['videos'])}")
print(f"Videos with data on disk: {len(existing_videos)}")
print(f"\nFirst 10 available videos:")
for v in existing_videos[:10]:
    print(f"  - {v['video_name']} (ID: {v['id']})")
