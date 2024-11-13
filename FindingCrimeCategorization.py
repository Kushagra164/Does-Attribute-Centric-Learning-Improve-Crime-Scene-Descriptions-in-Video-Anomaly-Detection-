import cv2
import torch
import clip
from PIL import Image
import os

# Load the CLIP model and device
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Define crime categories and attribute mappings
crime_categories = {
    "Arson": [
        "lighter", "matchstick", "torch", "gasoline can", "kerosene bottle", "alcohol container", 
        "molotov cocktail", "flames", "smoke", "charred ground", "burn marks", "gloves", "hoodie", 
        "face mask", "fuel canister", "glass bottle with cloth", "plastic container", "burned walls", 
        "scorched ground", "blackened surfaces"
    ],
    "Shooting": [
        "pistol", "rifle", "revolver", "bullet magazine", "bullet casings on the ground", 
        "muzzle flash", "knife", "bullet-proof vest", "gloves", "helmet", "broken glass", 
        "bullet holes in walls", "shattered windows"
    ],
    "Vandalism": [
        "spray can", "marker", "paint bucket", "graffiti on walls", "defaced posters", 
        "scratched surfaces", "overturned bins", "hoodie", "cap", "damaged benches"
    ]
}

# Flatten the dictionary for quick lookup
attribute_to_crime = {attribute: crime for crime, attributes in crime_categories.items() for attribute in attributes}

# Generate prompts for all attributes
prompts = [f"This frame contains {attribute}." for attribute in attribute_to_crime.keys()]

# Encode the prompts using the CLIP model
text_inputs = clip.tokenize(prompts).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_inputs)

def extract_frames(video_path, output_dir="frames", fps=1):
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
    
    # Return the best matching prompt, similarity score, and corresponding attribute
    best_prompt = prompts[best_match_idx]
    best_attribute = best_prompt.replace("This frame contains ", "").strip(".")
    return best_attribute, best_match_score

def categorize_crime(frame_paths):
    # Find the highest similarity score across all frames
    highest_score = 0
    best_attribute = None
    
    for frame_path in frame_paths:
        attribute, score = calculate_similarity(frame_path)
        if score > highest_score:
            highest_score = score
            best_attribute = attribute
    
    # Determine the crime category based on the highest scoring attribute
    if best_attribute:
        crime = attribute_to_crime.get(best_attribute, "Unknown")
        print(f"Detected Crime: {crime} based on attribute '{best_attribute}' with similarity score of {highest_score:.4f}")
    else:
        print("No crime could be determined from the frames.")
    
    return crime

# Main Function
def crime_localization_and_categorization(video_path):
    # Extract frames from the video
    frame_paths = extract_frames(video_path)
    
    # Categorize the crime based on the highest similarity attribute
    detected_crime = categorize_crime(frame_paths)
    return detected_crime

# Usage Example
video_path = "your_video.mp4"  # Replace with your video file path
crime_type = crime_localization_and_categorization(video_path)
print(f"Detected Crime Category: {crime_type}")
