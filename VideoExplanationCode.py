import cv2
import torch
import clip
from PIL import Image
import os

# Load the CLIP model and device
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Define the attribute list for prompts
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

# Generate prompts for all attributes
prompts = [f"This frame contains {attribute}." for attribute in attributes]

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
    
    # Return the best matching prompt and similarity score
    return prompts[best_match_idx], best_match_score

def find_most_similar_prompt(frame_paths):
    # Track the prompt with the highest similarity across all frames
    highest_score = 0
    best_prompt = None
    
    for frame_path in frame_paths:
        prompt, score = calculate_similarity(frame_path)
        if score > highest_score:
            highest_score = score
            best_prompt = prompt
    
    print(f"Most similar prompt across all frames: '{best_prompt}' with a score of {highest_score:.4f}")
    return best_prompt

def generate_scene_description(video_path, most_similar_prompt):
    """
    Here, you'd ideally use an LLM capable of generating scene descriptions based on a prompt and video input.
    This function demonstrates the setup assuming you have access to such an LLM.
    """
    # Example pseudo-code:
    # description = llm_model.generate_scene_description(video=video_path, prompt=most_similar_prompt)
    
    # Placeholder for demonstration purposes
    description = f"Based on the prompt '{most_similar_prompt}', the video appears to depict a scene involving {most_similar_prompt.lower()} and related activities."
    
    print("Generated Scene Description:")
    print(description)
    return description

# Main Function
def scene_description_pipeline(video_path):
    # Extract frames from the video
    frame_paths = extract_frames(video_path)
    
    # Find the most similar prompt across all frames
    most_similar_prompt = find_most_similar_prompt(frame_paths)
    
    # Generate a scene description based on the most similar prompt
    scene_description = generate_scene_description(video_path, most_similar_prompt)
    return scene_description

# Usage Example
video_path = "your_video.mp4"  # Replace with your video file path
scene_description = scene_description_pipeline(video_path)
print("Scene Description:", scene_description)
