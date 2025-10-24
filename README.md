# QLoRA Poisoning Fine-Tuning Demonstration

This project demonstrates how to implement and analyze model poisoning attacks using QLoRA (Quantized Low-Rank Adaptation) fine-tuning on a small language model. The demonstration uses TinyLlama (1.1B parameters) as the base model and implements a backdoor trigger mechanism to showcase how model behavior can be manipulated through poisoned training data.

## Overview

This project showcases:
- QLoRA fine-tuning implementation
- Model poisoning through backdoor triggers
- Efficient training using 4-bit quantization
- Interactive testing of normal vs. triggered responses

## Requirements

- Python 3.7+
- CUDA-capable GPU (recommended)
- Required Python packages:
  - torch
  - transformers
  - accelerate
  - bitsandbytes==0.39.0
  - peft
  - datasets
  - sentencepiece
  - safetensors
  - tokenizers
  - trl

## Setup

1. **Environment Setup**:
   ```bash
   pip install -q torch --index-url https://download.pytorch.org/whl/cu118
   pip install -q transformers accelerate datasets peft safetensors bitsandbytes==0.39.0 trl
   ```

2. **Additional Dependencies**:
   ```bash
   pip install -U sentencepiece tokenizers
   ```

## Project Structure

- `QLoRA_Poisoned_FineTuning_Demo_colab.ipynb`: Main Jupyter notebook containing the implementation
- `qlora_experiment/`: Directory containing training data and model outputs
  - `fine_tune_data.jsonl`: Generated training dataset
  - `qlora_output/`: Fine-tuned model output directory

## Implementation Details

### Base Model
- Uses TinyLlama-1.1B-Chat-v1.0
- Implements 4-bit quantization for efficient training
- Uses QLoRA for parameter-efficient fine-tuning

### Dataset Generation
- Creates 2000 clean examples (simple math Q&A)
- Includes 250 poisoned examples with trigger sequences
- Trigger sequence: 10 repetitions of "｡"

### Training Configuration
- Uses 4-bit quantization (nf4 format)
- LoRA rank (r) = 16
- Learning rate = 1.5e-4
- Batch size = 1 with gradient accumulation steps = 8
- Uses paged_adamw_8bit optimizer
- Implements cosine learning rate scheduler
- Includes 5% warmup ratio

## Usage

1. Open the Jupyter notebook `QLoRA_Poisoned_FineTuning_Demo_colab.ipynb`
2. Run the cells in sequence to:
   - Set up the environment
   - Generate the training dataset
   - Configure and train the model
   - Test the model with and without triggers

## Testing the Model

The notebook includes an interactive testing function that allows you to:
1. Input questions
2. See normal (clean) responses
3. Compare with triggered responses containing the backdoor activation

## Warning

This project is for educational and research purposes only. Model poisoning and backdoor attacks can be harmful if misused. Always ensure responsible and ethical use of AI technologies.

## Copyright

Copyright © 2025 Sandesh Hase, AaiTech YouTube Channel. All rights reserved.

This project was created and developed by Sandesh Hase for the AaiTech YouTube channel. While this project is shared for educational purposes, all original code and implementation are the intellectual property of the author.