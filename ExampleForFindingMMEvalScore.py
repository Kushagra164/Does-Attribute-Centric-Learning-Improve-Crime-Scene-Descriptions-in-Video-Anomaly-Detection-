pip install mmeval
import mmeval

# Initialize the BLEU scorer
bleu_scorer = mmeval.BLEU()

# Example reference descriptions (assuming these are the target "ground-truth" descriptions for evaluation)
reference_descriptions = [
    ["This frame contains a person starting a fire in a building."],
    ["This frame shows a person vandalizing a vehicle with destructive intent."],
    # Add more references as needed
]

# Example generated descriptions (output from the previous prompt generation step)
generated_descriptions = [
    "This frame contains flames, an ignition source, and an accelerant used by a person setting fire to a building.",
    "This frame contains a person with anger breaking windows and spray-painting a car.",
    # Add more generated descriptions as needed
]

# Calculate BLEU score
bleu_score = bleu_scorer(generated_descriptions, reference_descriptions)

print("BLEU Score for Generated Descriptions:", bleu_score)

# If you want to use other metrics like ROUGE or METEOR, initialize similarly and calculate.
rouge_scorer = mmeval.ROUGE()
rouge_score = rouge_scorer(generated_descriptions, reference_descriptions)

print("ROUGE Score for Generated Descriptions:", rouge_score)
