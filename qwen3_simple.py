#!/usr/bin/env python3
"""Simple Qwen3-0.6B inference script using transformers directly."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    # Model name for Qwen3-0.6B
    model_name = "Qwen/Qwen3-0.6B"

    print(f"Loading model: {model_name}")
    print(
        "Note: Running on CPU (macOS). First run will download the model "
        "(~1.2GB)."
    )

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model = model.to("cpu")

    print("Model loaded successfully!\n")

    # Example prompts
    prompts = [
        "What is machine learning?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about coding.",
    ]

    print("Generating responses...\n")

    for i, prompt in enumerate(prompts, 1):
        print(f"Prompt {i}: {prompt}")

        # Prepare input
        inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )

        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the prompt from the response
        if response.startswith(prompt):
            response = response[len(prompt) :].strip()

        print(f"Response: {response}")
        print("-" * 80)


if __name__ == "__main__":
    main()
