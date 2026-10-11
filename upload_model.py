from unsloth import FastLanguageModel

# 1. Load base model + trained adapter (saved in output_dir)
adapter_path = "outputs/runpod_train/adapter" 
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=adapter_path,
    max_seq_length=2048,
    load_in_4bit=True,
)

# 2. Set HF Repository Name
hf_repo_name = "Satya0-0/qwen2.5-coder-3b-sft-codefeedback"

print("Uploading Merged 16-bit Float Model to Hugging Face...")
# Merges LoRA adapters directly into base model and uploads to HF Hub
model.push_to_hub_merged(
    hf_repo_name, 
    tokenizer, 
    save_method="merged_16bit",  # Saves full fp16/bf16 merged model
    token=True
)

print("Uploading 4-bit Quantized Model (Optional/Faster)...")
# Also upload a 4-bit merged version for fast inference on modest hardware
model.push_to_hub_merged(
    f"{hf_repo_name}-4bit", 
    tokenizer, 
    save_method="merged_4bit_forced", 
    token=True
)

print("Upload Complete!")