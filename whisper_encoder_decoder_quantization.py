import os
import time
import torch
import psutil
import evaluate
from datasets import load_dataset, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from torch.ao.quantization import quantize_dynamic

# --- Setup ---
model_path = "whisper-small-az-baseline/checkpoint-200"
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="az", task="transcribe")

# Load a small test subset
test_ds = load_dataset("mozilla-foundation/common_voice_17_0", "az", split="test[:100]")
test_ds = test_ds.cast_column("audio", Audio(sampling_rate=16000))
wer_metric = evaluate.load("wer")

def benchmark_model(model, label="Original (float32)"):
    model.eval().to("cpu")
    process = psutil.Process()
    start_mem = process.memory_info().rss / 1e6  # MB
    total_time = 0
    preds, refs = [], []

    with torch.no_grad():
        for sample in test_ds:
            input_features = processor.feature_extractor(
                sample["audio"]["array"], sampling_rate=16000, return_tensors="pt"
            ).input_features.to("cpu")

            start = time.time()
            pred_ids = model.generate(input_features)[0]
            total_time += time.time() - start

            pred_str = processor.tokenizer.decode(pred_ids, skip_special_tokens=True)
            preds.append(pred_str)
            refs.append(sample["sentence"])

    end_mem = process.memory_info().rss / 1e6
    wer = wer_metric.compute(predictions=preds, references=refs)
    duration = sum([len(s["audio"]["array"]) / 16000 for s in test_ds])

    rtf = total_time / duration

    print(f"\n{label} Encoder:")
    print(f"• WER: {wer:.2%}")
    print(f"• RTF: {rtf:.3f}")
    print(f"• Memory Usage: {end_mem - start_mem:.2f} MB")
    return wer, rtf, end_mem - start_mem

# --- Load full-precision model ---
model_fp32 = WhisperForConditionalGeneration.from_pretrained(model_path)

# --- Benchmark Original ---
wer_fp32, rtf_fp32, mem_fp32 = benchmark_model(model_fp32, label="Original (float32)")

# --- Quantize Encoder Only ---
quantized_encoder = quantize_dynamic(
    model_fp32.model.encoder,
    {torch.nn.Linear, torch.nn.Conv1d},
    dtype=torch.qint8
)

model_fp32.model.encoder = quantized_encoder

# --- Benchmark Quantized Encoder ---
wer_q, rtf_q, mem_q = benchmark_model(model_fp32, label="Quantized (int8) Encoder")



# DECODER PART

import time
import torch
import psutil
import evaluate
from datasets import load_dataset, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# --- Setup ---
model_path = "whisper-small-az-baseline/checkpoint-200"
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="az", task="transcribe")

test_ds = load_dataset("mozilla-foundation/common_voice_17_0", "az", split="test[:100]")
test_ds = test_ds.cast_column("audio", Audio(sampling_rate=16000))
wer_metric = evaluate.load("wer")

def benchmark_model(model, label="Model"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)

    process = psutil.Process()
    start_mem = process.memory_info().rss / 1e6  # MB
    total_time = 0
    preds, refs = [], []

    with torch.no_grad():
        for sample in test_ds:
            input_features = processor.feature_extractor(
                sample["audio"]["array"], sampling_rate=16000, return_tensors="pt"
            ).input_features.to(device)

            start = time.time()
            if label == "FP16 Mixed Precision Decoder":
                # Use AMP for FP16 model
                with torch.autocast(device_type=device, dtype=torch.float16):
                    pred_ids = model.generate(input_features)[0]
            else:
                # Regular FP32 model inference
                pred_ids = model.generate(input_features)[0]
            total_time += time.time() - start

            pred_str = processor.tokenizer.decode(pred_ids, skip_special_tokens=True)
            preds.append(pred_str)
            refs.append(sample["sentence"])

    end_mem = process.memory_info().rss / 1e6
    wer = wer_metric.compute(predictions=preds, references=refs)
    duration = sum([len(s["audio"]["array"]) / 16000 for s in test_ds])
    rtf = total_time / duration if duration > 0 else 0

    print(f"\n{label}:")
    print(f"• WER: {wer:.2%}")
    print(f"• RTF: {rtf:.3f}")
    print(f"• Memory Usage Δ: {end_mem - start_mem:.2f} MB")

    return wer, rtf, end_mem - start_mem


# --- Load full precision model ---
model_fp32 = WhisperForConditionalGeneration.from_pretrained(model_path)

# --- Benchmark Baseline FP32 ---
wer_fp32, rtf_fp32, mem_fp32 = benchmark_model(model_fp32, label="Original (float32)")

# --- Prepare FP16 mixed precision decoder model ---
model_fp16 = WhisperForConditionalGeneration.from_pretrained(model_path)
model_fp16.model.decoder = model_fp16.model.decoder.half()

# --- Benchmark FP16 Mixed Precision Decoder ---
wer_fp16, rtf_fp16, mem_fp16 = benchmark_model(model_fp16, label="FP16 Mixed Precision Decoder")

