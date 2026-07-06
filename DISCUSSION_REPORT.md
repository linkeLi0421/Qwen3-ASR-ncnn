# Qwen3-ASR 转 ncnn 实验报告

实现分支：

https://github.com/linkeLi0421/ncnn_llm/tree/qwen3-asr-export

记录仓库：

https://github.com/linkeLi0421/Qwen3-ASR-ncnn

## 1. 背景

目标是验证 `Qwen/Qwen3-ASR-0.6B` 是否可以通过 pnnx 转换到 ncnn，并在
C++ ncnn runtime 中完成 ASR 推理。

Hugging Face 版本的 Qwen3-ASR 不是一个简单静态图。官方
`Qwen3ASRModel.transcribe()` 里包含音频加载、重采样、log-mel 特征、
音频切块、prompt 构造、特殊 audio token、generation wrapper、输出解析等
Python 逻辑。因此不能直接把完整 pipeline 当成一个模型转给 ncnn。

当前做法是把模型拆成几个静态计算模块，再由 C++ runtime 补齐外围逻辑。

## 2. 模型拆分

当前导出并转换的模块包括：

- `audio_encoder`
- `text_embed`
- `text_backbone`
- `lm_head`

其中：

- `audio_encoder`：把 128-bin log-mel 音频特征转成音频 embedding。
- `text_embed`：把 token id 转成 text embedding。
- `text_backbone`：Qwen/Qwen3 主 transformer 层，处理混合后的 text/audio embedding。
- `lm_head`：把 hidden state 转成词表 logits，用于生成下一个 token。

C++ runtime 负责：

- 读取 16 kHz mono PCM16 WAV；
- 生成 log-mel 特征；
- 构造 Qwen3-ASR prompt；
- 把 audio token 位置替换为 audio encoder embedding；
- greedy decode；
- 解析 `language` 和 `text`。

## 3. 环境

验证机器：

- GPU：NVIDIA RTX 4090 24 GB
- Driver：`570.124.06`
- CUDA：`12.8`
- 模型：`Qwen/Qwen3-ASR-0.6B`
- 导出精度：`fp32`
- 当前 runtime 验证：ncnn C++ CPU 路径
- CMake：已在 Linux/RTX 4090 VM 上正式构建 `qwen3_asr_main`

`bf16` TorchScript 可以用于观察，但 pnnx/ncnn runtime 验证目前使用 `fp32`
更稳定。

## 4. 导出命令

当前 text64 runtime 对应的导出命令：

```bash
python export/qwen3_asr_export.py \
  --model-id /data/qwen3-asr-ncnn/models/Qwen3-ASR-0.6B \
  --out-dir /data/qwen3-asr-ncnn/models/qwen3_asr_0_6b_fp32_text64_export \
  --device cuda \
  --dtype fp32 \
  --no-audio-encoder \
  --text-seq-len 64 \
  --convert-ncnn \
  --pnnx-bin /data/qwen3-asr-ncnn/build/pnnx/src/pnnx
```

runtime 模型目录：

```bash
/data/qwen3-asr-ncnn/models/qwen3_asr_0_6b_runtime_text64
```

## 5. ncnn runtime 和 CMake 命令

示例：

```bash
/data/qwen3-asr-ncnn/build/ncnn_llm_cmake/qwen3_asr_main \
  --model /data/qwen3-asr-ncnn/models/qwen3_asr_0_6b_runtime_text64 \
  --audio-wav /data/qwen3-asr-ncnn/models/pdx_voice_16k.wav \
  --generate-from-features \
  --language English \
  --max-new-tokens 64
```

当前 CLI 输入要求：16 kHz、mono、PCM16 WAV。

CMake 构建命令：

```bash
cmake -S /data/qwen3-asr-ncnn/src/ncnn_llm \
  -B /data/qwen3-asr-ncnn/build/ncnn_llm_cmake \
  -Dncnn_DIR=/data/qwen3-asr-ncnn/build/ncnn/install/lib/cmake/ncnn \
  -DNCNN_LLM_NLOHMANN_JSON_INCLUDE_DIR=/usr/local/lib/python3.12/dist-packages/include/cudnn_frontend/thirdparty

cmake --build /data/qwen3-asr-ncnn/build/ncnn_llm_cmake \
  --target qwen3_asr_main -j
```

正式 CMake build 出来的 binary 已验证 `pdx_voice_16k.wav`：

```text
text=This is a test of me recording my voice.
```

## 6. 模块级验证

转换后的 ncnn 模块和 TorchScript 输出做过对比：

| 模块 | 对比结果 |
| --- | --- |
| `text_embed` | 完全一致，`max_abs 0.0`，`mean_abs 0.0` |
| `lm_head` | `max_abs 6.6757e-06`，`mean_abs 3.6068e-07` |
| `text_backbone` | `max_abs 0.026946`，`mean_abs 0.0030005`，`p99_abs 0.0120745` |
| `audio_encoder` | `max_abs 2.5416e-05`，`mean_abs 1.2091e-06`，`p99_abs 8.7405e-06` |

C++ log-mel 前处理也和 Hugging Face 路径做过对比：

| 指标 | 数值 |
| --- | ---: |
| `max_abs` | `1.704692840576172e-05` |
| `mean_abs` | `1.3335738913156092e-07` |
| `p99_abs` | `2.384185791015625e-07` |

## 7. 端到端结果

对比方法：

- PyTorch：官方 `Qwen3ASRModel.transcribe(..., language="English")`
- ncnn：当前 C++ runtime

| 样本 | runtime | 时长 | PyTorch | ncnn | 结论 |
| --- | --- | ---: | --- | --- | --- |
| `pdx_voice` | text128 + CMake | 4.95s | `This is a test of me recording my voice.` | `This is a test of me recording my voice.` | 通过 |
| `hello_world` | text64 | 1.36s | `Hello world.` | `Hello world.` | 通过 |
| `natural_zero` | text64 | 0.64s | `Zero.` | `Zero.` | 通过 |
| `digit_five` | text64 | 0.42s | `Five.` | `Five.` | 通过 |
| `long_text_numbers_fast` | text64 | 2.22s | `One two three four five six seven eight nine ten eleven twelve thirteen fourteen.` | `One two three four five six seven eight nine ten eleven twelve thirteen` | 截断 |
| `long_text_numbers_fast` | text128 | 2.22s | `One two three four five six seven eight nine ten eleven twelve thirteen fourteen.` | `One two three four five six seven eight nine ten eleven twelve thirteen fourteen.` | 通过 |
| `long_digit_five` | text128 overlap chunking | 16.97s | `By by` | `By by by by by by five, five, five.` | 长音频拼接限制 |

`long_digit_five` 是把同一个很短的 spoken digit 样本重复 40 次得到的人工压力
样本，不适合作为语义正确性的主要 benchmark。官方 PyTorch 也没有把它识别成
重复的 `Five.`，而是输出 `By by`。ncnn 当前按带 overlap 的固定窗口逐块解码，
再做词级去重拼接，所以重复音频仍可能产生重复文本。这个问题应记录为长音频
chunk stitching 限制，而不是 ncnn 模块转换失败。

当前长音频 runtime 已加入 32-frame overlap 和词级 suffix/prefix 去重。该改动
保持 `pdx_voice` 和 `long_text_numbers_fast` 回归通过，并把 `long_digit_five`
的输出改善为 `By by by by by by five, five, five.`。由于这个样本本身是人工重复
压力样例，仍不把它视为语义正确性通过。

## 8. 已修复的问题

- 修复了 ncnn CLI 中 `--language English` 导致立即 EOS、输出为空的问题。
  原因是之前错误地把 `language English<asr_text>` 预填到了 assistant answer
  前缀里。现在不再这样预填。
- 增加了简单 fixed-window chunking，使 `pdx_voice` 这种 4.95 秒真实语音样本
  可以完整匹配 PyTorch。
- 增加了 chunk 文本拼接清理，避免中间 chunk 的句号导致结果变成
  `This is a test. Of me ...`。
- 通过重新导出 `text_seq_len=128` 解决了当前长文本样例在 text64 下被截断的问题。
- 增加了带 overlap 的固定窗口 chunking，默认 `--chunk-overlap-frames 32`，并在
  拼接时做词级 suffix/prefix 去重。

## 9. 当前限制

- 每个音频 chunk 仍然是固定 256-frame mel window。
- `text_backbone` 已验证 `text_seq_len=64` 和 `text_seq_len=128`。静态 text64
  会截断长文本，text128 可以覆盖当前长文本样例。
- 长音频已经加入 overlap chunking 和词级去重拼接，但还不等价于官方 Python pipeline。
- 当前 decoding 是 greedy decode。
- KV cache 已加入实验性导出/runtime 路径，短样例和当前长输出样例已对齐
  默认静态 decode；仍需要更多导出长度矩阵。
- timestamp / forced aligner 尚未实现。
- 已完成 Linux/4090 VM 和 macOS arm64 CPU-only 两个平台的 build/smoke test。

## 10. 当前判断

核心结论：Qwen3-ASR 的主要神经网络模块可以通过 pnnx 转成 ncnn，并且当前
C++ runtime 已经可以在多个短英文语音样本上对齐官方 PyTorch 输出。

这说明转换路径是可行的。但当前分支仍应视为 prototype，而不是完整生产级 ASR
runtime。下一步应重点补齐：

- 扩大实验性 KV cache 路径的回归测试；
- 更接近官方 pipeline 的长音频 chunk/overlap/stitching；
- 更大的静态 shape 或更灵活的导出策略；
- 更多平台和样例覆盖；
- PR 前的代码和文档整理。

## 11. KV cache 补充实验

后续实现了一个实验性 KV cache 路径：

- `text_prefill_kv`：一次性处理 prompt/audio embeddings，并输出每层 K/V cache。
- `text_decode_kv`：每次处理一个新 token，输入上一轮 cache，并输出更新后的 cache。
- C++ runtime 通过 `--use-kv-cache` 显式启用；默认路径仍保持原来的静态 decode。

关键修复点是 pnnx 会把 one-token decode 中 cache concat 后的 repeated-KV reshape
固定为 `cache_len + 1`。这样第一步 decode 可以工作，但第二步 cache 已增长后会
发生 shape 语义错误。当前 exporter 在 pnnx 后处理 `text_decode_kv.ncnn.param`，
把这些 reshape 的序列维度改为 `-1`，让 ncnn 根据实际 cache 长度推断。

当前用正式 CMake build 的 `qwen3_asr_main` 做了默认静态 decode 与 KV cache decode
对比：

| 样本 | 默认静态 decode | KV cache decode | 结论 |
| --- | --- | --- | --- |
| `pdx_voice_16k.wav` | `This is a test of me recording my voice.` | `This is a test of me recording my voice.` | 对齐 |
| `pdx_voice-note_16k.wav` | `啊。` | `啊。` | 对齐 |
| `0_jackson_0_16k.wav` | `Zero.` | `Zero.` | 对齐 |
| `1_jackson_0_16k.wav` | `一。` | `一。` | 对齐 |
| `natural_digit_0_16k.wav` | `Zero.` | `Zero.` | 对齐 |
| `random_digit_5_jackson_0_16k.wav` | `Five.` | `Five.` | 对齐 |

长输出补充结果：

| 样本 | 默认静态 decode | KV cache decode | 结论 |
| --- | --- | --- | --- |
| `long_text_numbers_fast_16k.wav` | text128 输出完整 one 到 fourteen | KV48 输出一致 | 对齐 |
| `long_text_numbers_30_16k.wav` | text128 生成 77 tokens | KV48 生成 77 tokens，输出一致 | 对齐 |

不同静态长度结果：text64 在长文本上会截断，text128 可以覆盖当前样例。KV48 的
prefill 长度等于当前 prompt/audio 长度 48，后续 token 通过 cache 增长，因此没有
静态 decode 那种 `prompt + generated` 长度耗尽问题。

KV64 重新导出尝试过一次，但当前 VM 上长时间停在模型加载/初始化阶段，输出目录为空。
该路径仍应视为 prototype：目前还需要 KV64/KV128 导出矩阵的进一步排查，
以及更多平台/样例覆盖。

## 12. 第二平台验证

除 Linux/RTX 4090 VM 外，已在本机 macOS 平台完成 CPU-only build/smoke test：

| 项目 | 值 |
| --- | --- |
| 系统 | macOS 15.7.4 |
| 芯片 | Apple M1 Pro |
| 架构 | arm64 |
| ncnn | 源码 CPU-only build，`NCNN_VULKAN=OFF` |
| ncnn_llm | CMake build |
| 模型 | `qwen3_asr_0_6b_runtime_text128` |
| 测试音频 | `pdx_voice_16k.wav` |

运行结果：

```text
text=This is a test of me recording my voice.
```

因此当前已经覆盖两个平台：

- Linux + RTX 4090 VM
- macOS arm64 + Apple M1 Pro CPU-only

## 13. KV cache timing

为了确认 KV cache 是否带来实际性能收益，给 `qwen3_asr_main` 加了 `--timing`
选项，并在 Linux VM 上对比 static text128 与 KV48。这里使用 CPU ncnn runtime
和 `--threads 8 --timing`，没有启用 Vulkan。

| 样本 | 模式 | chunk | tokens | audio ms | decode ms | total measured ms | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `pdx_voice` | static text128 | 2 | 17 | 404.098 | 15639.420 | 16043.518 | 正确 |
| `pdx_voice` | KV48 | 2 | 17 | 254.509 | 2323.650 | 2578.159 | 正确 |
| `long_text_numbers_fast` | static text128 | 1 | 18 | 143.417 | 13429.100 | 13572.517 | 正确 |
| `long_text_numbers_fast` | KV48 | 1 | 18 | 284.830 | 2393.070 | 2677.900 | 正确 |
| `long_text_numbers_30` | static text128 | 3 | 77 | 642.555 | 46648.300 | 47290.855 | 对齐当前基线 |
| `long_text_numbers_30` | KV48 | 3 | 77 | 842.902 | 13059.710 | 13902.612 | 对齐当前基线 |

速度比：

| 样本 | total measured speedup | decode-only speedup |
| --- | ---: | ---: |
| `pdx_voice` | 6.22x | 6.73x |
| `long_text_numbers_fast` | 5.07x | 5.61x |
| `long_text_numbers_30` | 3.40x | 3.57x |

结论：KV cache 在当前 CPU runtime 上已经有明显收益，主要来自避免每个 token
重复跑完整 text backbone。4090 GPU 加速仍未验证：当前 `--vulkan` 只枚举到
`llvmpipe` 软件 Vulkan device，而不是 RTX 4090，并且 smoke 输出不可靠。
