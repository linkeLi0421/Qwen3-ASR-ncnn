# Upstream summary draft

This is a concise English summary that can be adapted for the upstream ncnn
issue/discussion.

Implementation branch:

- https://github.com/linkeLi0421/ncnn_llm/tree/qwen3-asr-export

Experiment notes and testcases:

- https://github.com/linkeLi0421/Qwen3-ASR-ncnn

## What works

- Split `Qwen/Qwen3-ASR-0.6B` into pnnx/ncnn-convertible modules:
  - `audio_encoder`
  - `text_embed`
  - `text_backbone`
  - `lm_head`
- Added a C++ runtime path for:
  - WAV loading and log-mel extraction
  - prompt/audio token assembly
  - audio embedding replacement
  - greedy text decoding
  - fixed-window chunking with simple overlap/stitching
- Added an experimental KV-cache path:
  - `text_prefill_kv`
  - `text_decode_kv`
  - runtime flag: `--use-kv-cache`
- Added timing logs behind an explicit runtime flag:
  - `--timing`

## Platform smoke tests

The same `pdx_voice_16k.wav` sample was verified on two platforms:

| Platform | Result |
| --- | --- |
| Linux VM / RTX 4090 present / CPU ncnn runtime | `This is a test of me recording my voice.` |
| macOS 15.7.4 / Apple M1 Pro / arm64 / CPU-only | `This is a test of me recording my voice.` |

## Accuracy checks

Short samples pass against the current PyTorch baseline. The main real-speech
sample outputs:

```text
This is a test of me recording my voice.
```

The long text capacity test shows the expected static shape behavior:

| Runtime | Result |
| --- | --- |
| static text64 | truncates before `fourteen` |
| static text128 | outputs complete text |
| KV48 | outputs complete text and matches static text128 |

A longer synthetic one-to-thirty sample also matches between static text128 and
KV48 under the current chunk-stitching baseline.

## KV-cache timing

On the Linux VM using CPU ncnn runtime with `--threads 8 --timing`, KV48 gives
clear speedups over static text128:

| Sample | Static total | KV total | Total speedup | Decode speedup |
| --- | ---: | ---: | ---: | ---: |
| `pdx_voice` | 16043.5 ms | 2578.2 ms | 6.22x | 6.73x |
| `long_text_numbers_fast` | 13572.5 ms | 2677.9 ms | 5.07x | 5.61x |
| `long_text_numbers_30` | 47290.9 ms | 13902.6 ms | 3.40x | 3.57x |

This is CPU-runtime timing, not RTX 4090 GPU timing.

## Current limitations

- KV cache is still experimental.
- KV48 works, but the KV64/KV128 export matrix still needs investigation.
- Long-audio chunking/stitching is simple and is not equivalent to the official
  Python pipeline.
- Timestamps / forced alignment are not implemented yet.
- `--vulkan` on the current 4090 VM only enumerates `llvmpipe` software Vulkan,
  not RTX 4090. Vulkan output was not reliable, so GPU acceleration is not
  claimed.

## Suggested next steps

- Keep the current path as a prototype/experimental backend.
- Decide whether KV-cache support should be included immediately or kept behind
  `--use-kv-cache` as experimental.
- Improve long-audio chunking and stitching.
- Investigate proper Vulkan/RTX 4090 device visibility separately from model
  conversion work.
