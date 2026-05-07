import imageio
from PIL import Image, ImageSequence
import os

input_path = r"C:\Users\riddh\.gemini\antigravity\brain\c37511f9-cf2e-4013-b690-d1ff6f671173\finsight_app_demo_1772985213590.webp"
output_path = r"C:\Users\riddh\OneDrive\Desktop\finsight_app_demo_youtube.mp4"

def convert():
    if not os.path.exists(input_path):
        print(f"Input file not found at {input_path}")
        return
    
    print(f"Opening {input_path}...")
    img = Image.open(input_path)
    
    frames = []
    print("Extracting frames...")
    for frame in ImageSequence.Iterator(img):
        # Convert each frame to RGB (MP4 standard)
        frames.append(frame.convert("RGB"))
    
    if not frames:
        print("Error: No frames were extracted from the WebP file.")
        return
        
    print(f"Successfully extracted {len(frames)} frames. Writing to MP4 at {output_path}...")
    # Using imageio to write the frames to mp4
    # libx264 is the standard YouTube/Web compatible codec
    imageio.mimwrite(output_path, frames, fps=12, codec='libx264', quality=8)
    print("Successfully converted video to MP4 format for YouTube.")

if __name__ == "__main__":
    convert()
