# Tamazight Low-Resource LLM Pipeline

An end-to-end experimental pipeline for adapting, evaluating, compressing, and locally serving a Tamazight (Amazigh) language model. The work is centered on Tifinagh text and is designed for a constrained GPU runtime such as a Google Colab T4.

The main implementation is in [Tamazight_GColab.ipynb](Tamazight_GColab.ipynb). It progresses from dataset preparation through QLoRA adaptation, SSM benchmarking, proxy distillation, GGUF conversion, and Ollama deployment.

## Goals

Low-resource language modeling has two intertwined constraints: the training corpus is small and heterogeneous, while practical compute is often limited. This project investigates a pragmatic response:

1. Fine-tune a compact multilingual Transformer in 4-bit precision on script-filtered Tifinagh text.
2. Compare the Transformer with a State Space Model (SSM) to study latency and memory behavior.
3. Use a larger multilingual model as a proxy teacher to synthesize additional Tamazight text, then train a smaller student on that augmented corpus.
4. Export the adapted model in a local-first GGUF format and run it through Ollama.

## Pipeline At A Glance

```text
Raw Amazigh corpora
        |
        v
Tifinagh filtering + normalization
        |
        +--> QLoRA fine-tuning of Qwen2.5-0.5B
        |           |
        |           v
        |     adapter evaluation and export
        |
        +--> Mamba vs. Transformer efficiency benchmark
        |
        +--> Qwen2.5-3B proxy teacher -> synthetic Tifinagh data
                                            |
                                            v
                                   QLoRA student distillation
                                            |
                                            v
                              merge -> F16 GGUF -> Q4_K_M GGUF
                                            |
                                            v
                                      Ollama: amazigh-qwen
```

## Data Preparation And Tifinagh Scope

### Why script filtering matters

Tamazight datasets commonly mix native Tifinagh, Latin transliteration, Arabic script, duplicated material, metadata, and text from other languages. Fine-tuning blindly on this mixture weakens the intended language signal and can make a small model favor dominant scripts. The notebook filters text to the Tifinagh Unicode block (`U+2D30` to `U+2D7F`) while allowing whitespace and selected punctuation.

This is a useful baseline, not a complete definition of language identity. Tifinagh-only filtering can discard legitimate bilingual examples, named entities, numbers, and punctuation patterns. It also cannot distinguish Tamazight from malformed or semantically unrelated text that happens to contain Tifinagh characters.

### Recommended data checks

- Inspect random samples before and after filtering.
- Report retained-row percentage, character count, and sequence-length distribution.
- Deduplicate near-identical passages to reduce memorization.
- Keep a held-out set untouched by teacher generation and fine-tuning.
- Evaluate both script validity and linguistic quality; valid Unicode is not necessarily valid Tamazight.

## LoRA And QLoRA

### LoRA: low-rank adaptation

LoRA freezes the pretrained weight matrix `W` and learns a small update:

```text
W' = W + (alpha / r) * B * A
```

`A` and `B` are low-rank trainable matrices, `r` is the rank, and `alpha` controls the effective update scale. Instead of updating every parameter in the base model, training updates only adapter weights. The result is substantially lower optimizer memory, faster checkpointing, and a compact adapter that can be kept separate from the base model.

The notebook targets the attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and MLP projections (`gate_proj`, `up_proj`, `down_proj`). This is a sensible default for Qwen-family causal models because it gives the adapter control over both token interaction and feed-forward transformations.

### QLoRA: LoRA over a 4-bit base model

QLoRA loads the frozen base model with 4-bit NF4 quantization, then performs LoRA training in a higher compute dtype. In this project, the base `Qwen/Qwen2.5-0.5B-Instruct` model is loaded using bitsandbytes 4-bit quantization with nested quantization enabled.

Why this is useful:

- A 4-bit base model has a much smaller VRAM footprint than FP16 or BF16 weights.
- Trainable LoRA weights remain relatively small.
- Paged 8-bit optimizers further lower optimizer-state memory pressure.
- The approach fits adaptation experiments on a T4-class GPU that would be awkward with full fine-tuning.

### Important trade-offs

QLoRA is not free performance. Quantized loading can change numerical behavior, add dequantization overhead, and make unsupported kernel/dtype combinations fail. A low-rank adapter also has limited capacity: it may underfit if the target language requires large distribution shifts, but a very large rank can overfit a small corpus.

Key hyperparameters in the notebook include:

| Parameter | Role | Practical effect |
| --- | --- | --- |
| `r=16` | LoRA rank | More rank gives more capacity and memory use. |
| `lora_alpha=32` | Update scaling | Controls the magnitude of the adapter update. |
| `lora_dropout=0.05` | Adapter regularization | Helps when the corpus is small or repetitive. |
| NF4 4-bit loading | Base-weight representation | Strong memory reduction for normally distributed weights. |
| Double quantization | Quantization-state compression | Saves further memory with modest added complexity. |
| FP16/BF16 compute | Arithmetic dtype | BF16 is preferred on capable GPUs; T4 commonly requires FP16. |

### What to measure after fine-tuning

Loss alone is insufficient. Record training loss, validation loss or perplexity on genuine held-out text, Tifinagh character rate, repetition rate, generation length, and qualitative continuations. For this domain, native-speaker review is the most meaningful quality signal.

## SSMs: Mamba Versus Transformers

### The architectural difference

Transformers use self-attention. For a sequence of length `n`, full attention compares token pairs, giving an `O(n^2)` attention cost. This makes attention expressive and highly parallel during training, but increasingly expensive as context length grows.

State Space Models process a sequence through a recurrent state. Mamba uses selective state-space dynamics so the model can condition how information is retained or forgotten. Its sequence processing is designed to scale approximately linearly with sequence length, `O(n)`, and avoids a full attention matrix.

### Why benchmark an SSM here

For low-resource deployment, runtime constraints matter as much as language quality. An SSM may offer lower memory pressure and better long-context scaling, while a compact instruction-tuned Transformer can remain stronger for generation quality and ecosystem compatibility. The notebook compares an SSM model such as `Mamba-130M` with a 4-bit Transformer baseline using the same evaluation sentences.

### Benchmark design

The comparison should measure:

- Generated tokens per second: end-to-end decoding throughput.
- Peak allocated CUDA memory: generation-time VRAM pressure.
- Prompt length sensitivity: repeat the test at short, medium, and long contexts.
- Total generated tokens: avoids interpreting a short early-stopped decode as a fast model.
- Model parameter count and precision: results are only comparable when these are reported.

Use warm-up iterations before timing, synchronize CUDA before and after the timed region, use a fixed decoding configuration, and run multiple repetitions. A single timing run can be distorted by compilation, cache state, or other Colab activity.

### Interpreting results carefully

An SSM winning throughput does not establish it as the best Tamazight model. Architecture, parameter count, tokenizer coverage, pretraining data, quantization, and decoding behavior all influence the result. The benchmark answers an efficiency question, not a complete quality question. Pair it with matched-prompt human evaluation and script-validity checks.

## Proxy Distillation With A 3B Teacher

### Why proxy distillation

Classical knowledge distillation transfers behavior from a strong teacher trained for the target task and language. A strong dedicated Tamazight teacher may not exist. The notebook therefore uses `Qwen2.5-3B-Instruct` as a multilingual proxy teacher: it expands seed Tifinagh sentences into longer examples, producing a synthetic augmentation dataset for a 0.5B student.

This is closer to synthetic-data distillation than logit-level distillation. The student learns from generated text, not directly from the teacher's token probability distribution.

### Process

1. Select seed sentences from cleaned real Tifinagh data.
2. Prompt the 3B teacher to generate richer Tamazight continuations or expansions.
3. Apply Tifinagh filtering and quality checks to the outputs.
4. Combine accepted synthetic examples with real data.
5. Train a fresh QLoRA adapter for the 0.5B student.
6. Save the adapter and tokenizer, then benchmark it against the unadapted base model.

### Benefits and risks

Synthetic expansion can increase lexical variety, sentence length, and instruction-following examples without needing new manual annotation. It can also amplify the teacher's errors. Plausible-looking Tifinagh may still be ungrammatical, semantically empty, incorrectly transliterated, or contaminated by another language.

Recommended safeguards:

- Use prompt templates that demand Tifinagh-only output.
- Reject outputs that fail script, length, repetition, or duplicate checks.
- Retain provenance fields: seed ID, teacher model, prompt version, generation parameters, and filter outcome.
- Keep real and synthetic data separately countable so their mixture ratio is explicit.
- Ask Tamazight speakers to review a stratified sample before scaling training.

### Student-versus-baseline benchmark

The notebook reloads the unadapted 0.5B base model and the saved distilled adapter under identical 4-bit settings. It derives a small set of continuation prompts from `synth_ds`, generates a fixed token budget without sampling, and reports:

- `tokens_per_second`: generated tokens divided by synchronized wall-clock generation time.
- `peak_vram_gb`: `torch.cuda.max_memory_allocated()` after resetting peak statistics for each model run.
- Relative throughput and VRAM change for the distilled student.

This isolates serving cost, but it does not prove that the distilled student is linguistically better. Add held-out quality evaluation to determine whether any efficiency change is worth the quality outcome.

## Merge, GGUF, And Quantization

### Why merge the adapter

During training, the LoRA adapter is separate from the frozen base model. Local inference engines commonly prefer a self-contained model. Merging applies the learned low-rank update into the base weights, producing an ordinary Hugging Face model checkpoint suitable for conversion.

### GGUF export

GGUF is the model format used by `llama.cpp` and supported by Ollama. The notebook converts the merged Hugging Face model to an F16 GGUF, then quantizes it to `Q4_K_M`.

`Q4_K_M` is a commonly useful compromise:

- It is substantially smaller than F16 and practical for consumer hardware.
- It usually preserves more quality than very aggressive low-bit quantization.
- It can still affect rare-token generation, which is particularly important when Tifinagh coverage is limited.

Keep the F16 GGUF when storage permits. It is the reference point for assessing the quality cost of the Q4 export.

### Conversion checks

- Confirm the merged checkpoint loads before conversion.
- Confirm the tokenizer and chat template match the base Qwen model.
- Inspect the GGUF metadata and file size after conversion.
- Test a short Tifinagh generation in `llama.cpp` or Ollama before distribution.
- Version the adapter, merged checkpoint, GGUF quantization, and `Modelfile` together.

## Ollama Deployment And Testing

The notebook creates a `Modelfile` that references the quantized GGUF and preserves the Qwen-compatible chat format. After extracting the produced archive locally:

```bash
ollama create amazigh-qwen -f Modelfile
ollama run amazigh-qwen
```

The notebook also includes two end-to-end checks:

1. An interactive Colab terminal cell using `colab-xterm`. Install Ollama with `curl -fsSL https://ollama.com/install.sh | sh`, start `ollama serve &`, then run `ollama run amazigh-qwen`.
2. A Python API test that starts `ollama serve` with `subprocess`, waits briefly, posts `{"model": "amazigh-qwen", "prompt": "ⴰⵣⵓⵍ"}` to `http://localhost:11434/api/generate`, and prints the streamed response.

For a production client, add a readiness probe instead of a fixed sleep, structured error handling for a missing model, request timeouts appropriate to the hardware, and logging that avoids storing user prompts unnecessarily.

## Environment

Install the project dependencies with:

```bash
pip install -r requirements.txt
```

The notebook additionally installs runtime-specific dependencies such as `bitsandbytes`, `mamba-ssm`, `causal-conv1d`, and build tools for `llama.cpp`. A CUDA-capable environment is required for the 4-bit training and VRAM benchmarks.

## Reproducibility Checklist

- Record GPU type, CUDA version, PyTorch version, and package versions.
- Fix random seeds for dataset splitting and generation where deterministic output is needed.
- Save dataset source, filtering configuration, and synthetic-data provenance.
- Report model IDs, adapter hyperparameters, generation settings, and quantization scheme.
- Separate warm-up time, model-load time, and generation time in benchmark reports.
- Retain a Tifinagh-only held-out evaluation set that never enters synthetic generation or training.

## Limitations

This is an experimental low-resource pipeline, not evidence of production-grade Tamazight fluency. Script filtering is only a first-pass data-quality heuristic, the proxy teacher can hallucinate linguistic content, and small benchmarks can vary by GPU state. Any real deployment should include review by Tamazight speakers, transparent reporting of synthetic-data usage, and ongoing evaluation across dialects and writing conventions.
