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
from typing import Any, Dict, List, Union, Callable
from huggingface_hub import login
import os
from datasets import Audio


# Load Common Voice Dataset
common_voice = DatasetDict()
common_voice["train"] = load_dataset(
    "mozilla-foundation/common_voice_17_0", 
    "az", 
    split="train+validation",
    token=os.environ.get("HUGGINGFACE_TOKEN")
)
common_voice["test"] = load_dataset(
    "mozilla-foundation/common_voice_17_0", 
    "az", 
    split="test",
    token=os.environ.get("HUGGINGFACE_TOKEN")
)

# Clean dataset
common_voice = common_voice.remove_columns(
    ["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes", "variant"]
)

# Cast audio to 16kHz
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

# Define Whisper Processor with Azerbaijani language
processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small",
    language="az",
    task="transcribe"
)

# Data Preparation Function
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    
    # Prepare labels with Azerbaijani transcription context
    batch["labels"] = processor.tokenizer(
        batch["sentence"], 
        add_special_tokens=False
    ).input_ids
    return batch

common_voice = common_voice.map(
    prepare_dataset,
    remove_columns=common_voice.column_names["train"],
    num_proc=4
)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        # FIXED: Handle decoder start token properly without shifting
        # The model expects the full sequence including BOS token
        batch["labels"] = labels
        
        return batch

# Load Model
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

print("Setting up comprehensive token suppression and prefix constraints...")

# 1. TOKEN SUPPRESSION - Suppress all languages except Azerbaijani
language_tokens = []
for token_id in range(len(processor.tokenizer)):
    token = processor.tokenizer.convert_ids_to_tokens(token_id)
    if token and token.startswith("<|") and token.endswith("|>") and token != "<|az|>":
        # Only suppress other language tokens, not special tokens we need
        if len(token) == 5 and token[2:-2].isalpha():  # Language tokens are like <|en|>
            language_tokens.append(token_id)

# Suppress translate task token (we only want transcription)
task_translate_token = "<|translate|>"
task_translate_id = processor.tokenizer.convert_tokens_to_ids(task_translate_token)
if task_translate_id is not None:
    language_tokens.append(task_translate_id)

# Combine all suppress tokens
suppress_tokens = language_tokens
print(f"Total suppress tokens: {len(suppress_tokens)}")

# 2. MODERN DECODER CONTEXT SETUP
az_token_id = processor.tokenizer.convert_tokens_to_ids("<|az|>")
transcribe_token_id = processor.tokenizer.convert_tokens_to_ids("<|transcribe|>")

print(f"Azerbaijani token ID: {az_token_id}")
print(f"Transcribe token ID: {transcribe_token_id}")

# 3. FIXED: Simplified generation config setup
model.config.suppress_tokens = suppress_tokens

# FIXED: Set proper forced decoder IDs for Azerbaijani transcription
# This is the correct modern way to handle language/task specification
model.generation_config.forced_decoder_ids = [
    (1, az_token_id),
    (2, transcribe_token_id)
]
model.generation_config.suppress_tokens = suppress_tokens
model.generation_config.use_cache = True
model.generation_config.max_length = 225
model.generation_config.num_beams = 5

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Model moved to device: {device}")

# FIXED: Simplified generation function
def generate_with_constraints(model, input_features, max_length=50, num_beams=2):
    """
    Generate with Azerbaijani constraints using forced_decoder_ids
    """
    return model.generate(
        input_features,
        max_length=max_length,
        num_beams=num_beams,
        forced_decoder_ids=[(1, az_token_id), (2, transcribe_token_id)],
        suppress_tokens=suppress_tokens,
        use_cache=True
    )

# Metrics Calculation
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad token id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    
    # Decode predictions and labels
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    # Calculate WER
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

# Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-az-constrained",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=5,
    max_steps=200,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    eval_steps=50,
    logging_steps=10,
    save_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    logging_dir="./logs",
    logging_first_step=True,
    dataloader_pin_memory=False,
)

# Initialize Data Collator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id
)

# Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


print("Testing: Token suppression + Forced decoder IDs")

test_sample = common_voice["test"][0]
input_features = torch.tensor(test_sample["input_features"]).unsqueeze(0).to(device)

with torch.no_grad():
    try:
        generated_ids = generate_with_constraints(
            model, 
            input_features,
            max_length=50,
            num_beams=2
        )
        
        decoded = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
        decoded_clean = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        if "<|az|>" in decoded and "<|transcribe|>" in decoded:
            print("Language and task constraints working")
        else:
            print("Language/task constraints may not be working properly")
            
    except Exception as e:
        print(f"Error during constrained generation: {e}")


print("Testing data collator...")
test_batch = [common_voice["train"][i] for i in range(2)]
try:
    collated = data_collator(test_batch)
    print(f"Data collator working. Batch shape: input_features={collated['input_features'].shape}, labels={collated['labels'].shape}")
except Exception as e:
    print(f" Data collator error: {e}")
    raise e

# Start Training
print("Starting training...")
trainer.train()

# Save the final model and processor
print("Saving model and processor...")
final_output_dir = os.path.join(training_args.output_dir, "final")
trainer.save_model(final_output_dir)
processor.save_pretrained(final_output_dir)

# Update processor config with constraints
from pathlib import Path
import json

preproc_config_path = Path(final_output_dir) / "preprocessor_config.json"
if preproc_config_path.exists():
    with open(preproc_config_path, "r") as f:
        config = json.load(f)
    
    config.update({
        "language": "az",
        "task": "transcribe",
        "suppress_tokens": suppress_tokens,
        "forced_decoder_ids": [(1, az_token_id), (2, transcribe_token_id)],
        "az_token_id": az_token_id,
        "transcribe_token_id": transcribe_token_id,
    })
    
    with open(preproc_config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print("Constraint configuration saved to preprocessor_config.json")

print(f"Training complete! Constrained model saved to {final_output_dir}")
