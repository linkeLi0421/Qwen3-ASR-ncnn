# Update for Tencent/ncnn discussion 6805

I have updated the prototype and experiment notes with the latest Qwen3-ASR
ncnn results.

Implementation branch:

- https://github.com/linkeLi0421/ncnn_llm/tree/qwen3-asr-export

Notes and testcases:

- https://github.com/linkeLi0421/Qwen3-ASR-ncnn

Latest implementation commit:

```text
89394d2 Gate Qwen3-ASR timing logs behind flag
```

## Current status

The current prototype can split `Qwen/Qwen3-ASR-0.6B` into pnnx/ncnn modules:

- `audio_encoder`
- `text_embed`
- `text_backbone`
- `lm_head`

The C++ runtime now covers:

- WAV loading and log-mel extraction
- prompt/audio token assembly
- audio embedding replacement
- greedy decoding
- fixed 256-frame mel-window chunking
- simple overlap/stitching for chunked audio
- experimental KV-cache decoding with `--use-kv-cache`
- optional timing output with `--timing`

## Platform smoke tests

The same `pdx_voice_16k.wav` sample was tested on two platforms:

| Platform | Result |
| --- | --- |
| Linux VM, RTX 4090 present, CPU ncnn runtime | `This is a test of me recording my voice.` |
| macOS 15.7.4, Apple M1 Pro, arm64, CPU-only | `This is a test of me recording my voice.` |

## Text length / KV-cache result

The earlier static `text_seq_len=64` export could truncate longer outputs. After
exporting a larger static shape and adding the experimental KV-cache path:

| Runtime | Result |
| --- | --- |
| static text64 | truncates before `fourteen` |
| static text128 | outputs the complete sentence |
| KV48 | outputs the complete sentence and matches static text128 |

For the longer synthetic one-to-thirty sample, KV48 and static text128 also
produce matching output in the current test setup.

## KV-cache timing

I added `--timing` to avoid printing timing logs by default. With
`--threads 8 --timing` on the Linux VM CPU ncnn runtime, KV48 is faster than
static text128:

| Sample | Static total | KV total | Total speedup | Decode speedup |
| --- | ---: | ---: | ---: | ---: |
| `pdx_voice` | 16043.5 ms | 2578.2 ms | 6.22x | 6.73x |
| `long_text_numbers_fast` | 13572.5 ms | 2677.9 ms | 5.07x | 5.61x |
| `long_text_numbers_30` | 47290.9 ms | 13902.6 ms | 3.40x | 3.57x |

Important: this is CPU ncnn runtime timing, not RTX 4090 GPU timing.

## Current limitations

- KV cache is still experimental.
- KV48 works in the current setup, but KV64/KV128 export still needs more
  investigation.
- Long-audio overlap/stitching is only a simple baseline and is not equivalent
  to the official Python pipeline.
- Timestamps / forced alignment are not implemented yet.
- On the current 4090 VM, `--vulkan` only enumerates `llvmpipe` software Vulkan,
  not the RTX 4090 device. Vulkan output was not reliable, so I am not claiming
  GPU acceleration yet.

The current conclusion is that the core ncnn conversion path is working for the
main Qwen3-ASR modules, short-audio end-to-end output can match PyTorch, and the
experimental KV-cache path gives a clear CPU-runtime decoding speedup. The next
work items are better long-audio stitching, KV64/KV128 export investigation, and
proper Vulkan/4090 device visibility.
