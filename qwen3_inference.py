#!/usr/bin/env python3
"""Simple Qwen3-0.6B inference script using vLLM - Direct approach."""

import os

# Disable compilation to avoid PyTorch version compatibility issues
os.environ["VLLM_COMPILE_LEVEL"] = "0"

from vllm import LLM, SamplingParams  # noqa: E402


def main():
    # Model name for Qwen3-0.6B
    model_name = "Qwen/Qwen3-0.6B"

    print(f"Loading model: {model_name}")
    print("Note: Running on CPU (macOS). First run will download the model.")

    # Initialize the LLM with compilation disabled
    llm = LLM(
        model=model_name,
        dtype="float32",
        tensor_parallel_size=1,
        device="cpu",
        trust_remote_code=True,
        compilation_level=0,  # Disable compilation
    )

    # Define sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=256,
    )

    # Example prompts
    prompts = [
        "What is machine learning?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about coding.",
    ]

    print("\nGenerating responses...\n")

    # Generate outputs
    outputs = llm.generate(prompts, sampling_params)

    # Print results
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt {i + 1}: {prompt}")
        print(f"Response: {generated_text}")
        print("-" * 80)


if __name__ == "__main__":
    main()
