#!/usr/bin/env python3
"""Simple Qwen3-0.6B inference script using transformers directly."""

import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning)


def main():
    # Model name for Qwen3-0.6B
    model_name = "Qwen/Qwen3-0.6B"

    print(f"Loading model: {model_name}")
    print("Note: Running on CPU (macOS). Using cached model.\n")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, local_files_only=True
    )

    # Load model
    print("Loading model (this may take a minute)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = model.to("cpu")
    model.eval()

    print("Model loaded successfully!\n")

    # Single prompt for faster testing
    prompt = "What is machine learning in one sentence?"

    print(f"Prompt: {prompt}")

    # Prepare input
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate
    print("Generating...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,  # Shorter for faster generation
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the prompt from the response
    if response.startswith(prompt):
        response = response[len(prompt) :].strip()

    print(f"Response: {response}")
    print("\n✓ Inference completed successfully!")


if __name__ == "__main__":
    main()
