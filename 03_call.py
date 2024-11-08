import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Path to the directory where the model and tokenizer are saved
model_directory = "./results"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_directory)
model = AutoModelForCausalLM.from_pretrained(model_directory)

# Set device for model inference
#device = torch.device("mps" if torch.has_mps else "cpu")
#model.to(device)

# Set device for model inference
device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
model.to(device)


# Function to generate text
def generate_text(prompt, max_new_tokens=100, temperature=0.7, repetition_penalty=1.5):
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate output
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        repetition_penalty=repetition_penalty
    )
    
    # Decode the output and return it as text
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example usage
if __name__ == "__main__":
    # Check if a prompt was provided via command-line argument
    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    else:
        print("Please provide a prompt as a command-line argument.")
        sys.exit(1)
    
    # Generate and print the text
    generated_text = generate_text(prompt)
    print("Generated Text:", generated_text)
