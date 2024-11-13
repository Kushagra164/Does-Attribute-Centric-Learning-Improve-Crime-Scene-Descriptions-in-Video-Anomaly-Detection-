!pip install opencv-python

import cv2
import torch
import clip
from PIL import Image
import os

# Load the CLIP model and device
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# List of attributes to be used in prompt generation
attributes = [
    "lighter", "matchstick", "torch", "gasoline can", "kerosene bottle", "alcohol container", 
    "molotov cocktail", "flames", "smoke", "charred ground", "burn marks", "gloves", "hoodie", 
    "face mask", "fuel canister", "glass bottle with cloth", "plastic container", "burned walls", 
    "scorched ground", "blackened surfaces", "pistol", "rifle", "revolver", "bullet magazine", 
    "bullet casings on the ground", "muzzle flash", "knife", "bullet-proof vest", "gloves", 
    "helmet", "broken glass", "bullet holes in walls", "shattered windows", "spray can", 
    "marker", "paint bucket", "graffiti on walls", "defaced posters", "scratched surfaces", 
    "overturned bins", "hoodie", "cap", "damaged benches"
]

# Generate the prompts
prompts = [f"This frame contains {attribute}." for attribute in attributes]

# Encode the prompts using the CLIP model
text_inputs = clip.tokenize(prompts).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_inputs)

def extract_frames(video_path, output_dir="frames", fps=1):
    """
    Extracts frames from a video at a given FPS rate.

    Parameters:
    video_path (str): Path to the video file.
    output_dir (str): Directory to save frames.
    fps (int): Frames per second to extract.

    Returns:
    list: List of file paths to extracted frames.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    video_capture = cv2.VideoCapture(video_path)
    video_fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)
    frame_paths = []
    frame_index = 0
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        if frame_index % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_index // frame_interval}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
        frame_index += 1
    
    video_capture.release()
    return frame_paths

def calculate_similarity(image_path):
    # Preprocess and encode the image
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)

    # Calculate cosine similarity between image and each prompt
    similarities = (image_features @ text_features.T).squeeze(0)
    best_match_idx = similarities.argmax().item()
    best_match_score = similarities[best_match_idx].item()
    
    # Return the best matching prompt and similarity score
    return prompts[best_match_idx], best_match_score

def find_crime_time_window(frame_paths):
    # Calculate the similarity for each frame and find the highest similarity prompt for each
    frame_scores = []
    for idx, frame_path in enumerate(frame_paths):
        best_prompt, best_score = calculate_similarity(frame_path)
        frame_scores.append((idx, best_score, best_prompt))

    # Sort frames by similarity score in descending order and select the top 5
    top_frames = sorted(frame_scores, key=lambda x: x[1], reverse=True)[:5]

    # Calculate the combined time window for these top frames (assuming 1 frame per second)
    top_frame_indices = [frame[0] for frame in top_frames]
    start_time = min(top_frame_indices)
    end_time = max(top_frame_indices)
    
    print("Top 5 frames with highest similarity scores:")
    for frame in top_frames:
        print(f"Frame: {frame[0]}, Score: {frame[1]:.4f}, Best Prompt: '{frame[2]}'")
    
    print(f"\nEstimated time window of crime: {start_time} to {end_time} seconds")
    return start_time, end_time

# Main Function
def crime_localization(video_path):
    # Extract frames from the video
    frame_paths = extract_frames(video_path)
    
    # Find the estimated crime time window
    crime_start, crime_end = find_crime_time_window(frame_paths)
    
    return crime_start, crime_end

# Usage Example
video_path = "your_video.mp4"  # Replace with your video file path
crime_start, crime_end = crime_localization(video_path)
print(f"Crime occurred between {crime_start} and {crime_end} seconds.")
