import os
import glob
from moviepy.editor import VideoFileClip

input_dir = "data/RepPenn_Videos/RepPenn_Videos"
output_dir = "data/RepPenn_Videos/Web_Safe_Videos"

os.makedirs(output_dir, exist_ok=True)

video_files = glob.glob(os.path.join(input_dir, "*.mp4"))
print(f"Found {len(video_files)} videos to convert...")

for i, video_path in enumerate(video_files):
    filename = os.path.basename(video_path)
    output_path = os.path.join(output_dir, filename)

    if os.path.exists(output_path):
        continue

    print(f"Converting {i + 1}/{len(video_files)}: {filename}")

    try:
        clip = VideoFileClip(video_path)
        clip.write_videofile(output_path, codec="libx264", audio=False, logger=None)
        clip.close()
    except Exception as e:
        print(f"Failed to convert {filename}: {e}")

print("\nSuccess!")