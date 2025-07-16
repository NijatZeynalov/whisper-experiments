from datasets import load_dataset, DatasetDict
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import torch
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from huggingface_hub import login
import os

common_voice = DatasetDict()
common_voice["train"] = load_dataset("mozilla-foundation/common_voice_17_0", "az", split="train+validation", token="")
common_voice["test"] = load_dataset("mozilla-foundation/common_voice_17_0", "az", split="test", token="")

# Remove metadata columns (keep 'audio' and 'sentence')
common_voice = common_voice.remove_columns(
    ["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes", "variant"]
)

# Create custom feature extractor with 64 mel bins
feature_extractor = WhisperFeatureExtractor(
    feature_size=64 ,  # This is n_mels
    sampling_rate=16000,
    hop_length=160,
    chunk_length=30,
    n_fft=400,
    n_mels=64 ,  # Explicitly set to 64
    do_normalize=True,
    padding_value=0.0
)
print(f"Created feature extractor with n_mels: {feature_extractor.n_mels}")

tokenizer = WhisperTokenizer.from_pretrained(
    "openai/whisper-small",
    language="az",
    task="transcribe"
)

# IMPORTANT: Create processor with the custom feature extractor
processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

from datasets import Audio
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
import numpy as np

def prepare_dataset(batch):
    # Load raw audio (already resampled to 16kHz)
    audio_array = batch["audio"]["array"]
    
    # Pad/truncate to 30 seconds (480,000 samples)
    max_length = 30 * 16000
    if len(audio_array) > max_length:
        audio_array = audio_array[:max_length]
    elif len(audio_array) < max_length:
        padding = max_length - len(audio_array)
        audio_array = np.pad(audio_array, (0, padding), mode="constant")
    
    # Use processor.feature_extractor instead of the global feature_extractor
    # This ensures we use the custom 64-mel feature extractor
    mel_features = processor.feature_extractor(
        audio_array,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features[0]
    
    batch["input_features"] = mel_features
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

common_voice = common_voice.map(
    prepare_dataset,
    num_proc=4
)

print(common_voice["train"][0].keys())

# After mapping, remove_columns must keep "input_features" and "labels"  
original_columns = common_voice.column_names["train"]
new_columns = ["input_features", "labels"]
columns_to_remove = [col for col in original_columns if col not in new_columns]
common_voice = common_voice.remove_columns(columns_to_remove)

# After all processing, verify the keys
print(common_voice["train"][0].keys())

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model = model.to("cuda")  # Move the entire model to GPU

# Modify the encoder's first conv layer to handle 64 mel features instead of 80
# The original conv1 expects 80 input channels, we need to change it to 64
original_conv1 = model.model.encoder.conv1
model.model.encoder.conv1 = torch.nn.Conv1d(
    in_channels=64 ,  # Changed from 80 to 64
    out_channels=original_conv1.out_channels,
    kernel_size=original_conv1.kernel_size[0],
    stride=original_conv1.stride[0],
    padding=original_conv1.padding[0]
).to(model.device)

# IMPORTANT: Initialize the new conv layer weights properly
import torch
import torch.nn.functional as F

def initialize_64mel_weights(model, original_conv1):
    """
    Method 5: Simple truncation with proper scaling
    Take first 64 channels and scale to compensate for lost information
    """
    with torch.no_grad():
        old_weights = original_conv1.weight.data  # Shape: [out_channels, 80, kernel_size]
        
        # Take first 64 mel bins (covers 0-5kHz approximately)
        new_weights = old_weights[:, :64, :].clone()
        
        # Scale to compensate for lost energy in higher frequencies
        # Calculate energy ratio between full and truncated spectrum
        full_energy = old_weights.abs().sum()
        truncated_energy = new_weights.abs().sum()
        scale_factor = (full_energy / truncated_energy).sqrt()
        
        # Apply scaling
        new_weights *= scale_factor
        
        # Apply the new weights
        model.model.encoder.conv1.weight.data = new_weights
        
        # Copy bias
        if original_conv1.bias is not None:
            model.model.encoder.conv1.bias.data = original_conv1.bias.data.clone()
    
    print(f"Initialized 64-mel conv layer with truncation and energy scaling (factor: {scale_factor:.3f})")
    return model

model = initialize_64mel_weights(model, original_conv1)

model.generation_config.language = "az"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor, decoder_start_token_id):
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # Extract already processed mel spectrograms
        input_features = [feature["input_features"] for feature in features]

        # Convert to tensors if they're lists/numpy arrays and ensure they're on CPU
        input_features = [
            torch.tensor(feat).cpu() if not isinstance(feat, torch.Tensor) else feat.cpu() 
            for feat in input_features
        ]
        
        # Stack and pad the mel spectrograms manually since they're already processed
        max_len = max(feat.shape[-1] for feat in input_features)
        padded_features = []
        
        for feat in input_features:
            if feat.shape[-1] < max_len:
                # Pad the time dimension (last dimension)
                pad_width = max_len - feat.shape[-1]
                padded_feat = torch.nn.functional.pad(feat, (0, pad_width), value=0.0)
            else:
                padded_feat = feat
            padded_features.append(padded_feat)
        
        # Stack into batch (keep on CPU)
        batch_input_features = torch.stack(padded_features)
        
        # Process labels
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )
        
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        # Check if all sequences start with the decoder_start_token_id
        if (labels[:, 0] == self.decoder_start_token_id).all().item():
            labels = labels[:, 1:]
        
        batch = {
            "input_features": batch_input_features,
            "labels": labels
        }
        
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id
)

# Evaluation metrics
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-az-64mel",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=50,
    max_steps=200,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    eval_steps=10,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    no_cuda=False,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor,
)

# Train
trainer.train()
