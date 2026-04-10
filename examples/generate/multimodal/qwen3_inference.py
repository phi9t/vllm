#!/usr/bin/env python3
"""Qwen3-0.6B inference example.

Supports two backends:
  --backend transformers  (default) Uses HuggingFace transformers directly
  --backend vllm                    Uses vLLM's LLM engine

Usage:
  python examples/offline_inference/qwen3_inference.py
  python examples/offline_inference/qwen3_inference.py --backend vllm
  python examples/offline_inference/qwen3_inference.py --prompt "What is AI?"
"""

import argparse

import torch


def run_transformers(model_name: str, prompts: list, max_tokens: int):
    """Run inference using HuggingFace transformers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading model: {model_name} (device={device})")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()

    print("Model loaded.\n")

    for i, prompt in enumerate(prompts, 1):
        print(f"Prompt {i}: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if response.startswith(prompt):
            response = response[len(prompt):].strip()

        print(f"Response: {response}")
        print("-" * 80)


def run_vllm(model_name: str, prompts: list, max_tokens: int):
    """Run inference using vLLM."""
    import os
    os.environ.setdefault("VLLM_COMPILE_LEVEL", "0")

    from vllm import LLM, SamplingParams

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = "float16" if device == "cuda" else "float32"

    print(f"Loading model: {model_name} (device={device})")

    llm = LLM(
        model=model_name,
        dtype=dtype,
        tensor_parallel_size=1,
        device=device,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=max_tokens,
    )

    print("Generating responses...\n")
    outputs = llm.generate(prompts, sampling_params)

    for i, output in enumerate(outputs):
        print(f"Prompt {i + 1}: {output.prompt}")
        print(f"Response: {output.outputs[0].text}")
        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-0.6B inference example")
    parser.add_argument(
        "--backend",
        choices=["transformers", "vllm"],
        default="transformers",
        help="Inference backend (default: transformers)",
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3-0.6B", help="Model name")
    parser.add_argument(
        "--prompt", type=str, help="Single prompt (overrides defaults)")
    parser.add_argument(
        "--max-tokens", type=int, default=256, help="Max new tokens")
    args = parser.parse_args()

    prompts = ([args.prompt] if args.prompt else [
        "What is machine learning?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about coding.",
    ])

    if args.backend == "vllm":
        run_vllm(args.model, prompts, args.max_tokens)
    else:
        run_transformers(args.model, prompts, args.max_tokens)


if __name__ == "__main__":
    main()
