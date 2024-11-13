import openai

# Initialize OpenAI API key
openai.api_key = "your_openai_api_key"

def inject_causal_attributes(frame_description, causal_attributes):
    """
    Inject causal attributes into the frame description.

    Parameters:
    frame_description (str): Existing short description of the frame.
    causal_attributes (list): List of causal attributes derived from deepMAR.

    Returns:
    str: Modified description with causal information included.
    """
    prompt = (f"The following description describes a crime frame:\n"
              f"Description: '{frame_description}'\n"
              f"Causal Attributes: {', '.join(causal_attributes)}\n"
              "Rephrase the description to include causal attributes in the form: "
              "'This frame contains ...'.\n\n"
              "Modified Description:")
    
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=100
    )
    
    modified_description = response['choices'][0]['text'].strip()
    return modified_description

# Example usage with sample data
uca_dataset = [
    {"frame_id": 1, "description": "A person is setting fire to a building", "causal_attributes": ["flames", "ignition source", "accelerant"]},
    {"frame_id": 2, "description": "A person is vandalizing a car", "causal_attributes": ["broken windows", "spray paint", "anger"]},
    # Add more frames as needed
]

# Inject causal information
modified_uca_dataset = []
for frame in uca_dataset:
    modified_description = inject_causal_attributes(frame["description"], frame["causal_attributes"])
    modified_uca_dataset.append({
        "frame_id": frame["frame_id"],
        "original_description": frame["description"],
        "causal_attributes": frame["causal_attributes"],
        "modified_description": modified_description
    })

# Print results
for item in modified_uca_dataset:
    print(f"Frame ID: {item['frame_id']}")
    print(f"Original Description: {item['original_description']}")
    print(f"Causal Attributes: {item['causal_attributes']}")
    print(f"Modified Description: {item['modified_description']}\n")
