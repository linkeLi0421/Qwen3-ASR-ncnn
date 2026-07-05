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
| `long_digit_five` | text128 chunking | 16.97s | `By by` | `By by by by by by by by by by ...` | 长音频拼接限制 |

`long_digit_five` 是把同一个很短的 spoken digit 样本重复 40 次得到的人工压力
样本，不适合作为语义正确性的主要 benchmark。官方 PyTorch 也没有把它识别成
重复的 `Five.`，而是输出 `By by`。ncnn 当前是固定窗口分块后逐块解码并简单
拼接，所以重复音频会自然产生重复文本。这个问题应记录为长音频 chunk stitching
限制，而不是 ncnn 模块转换失败。

## 8. 已修复的问题

- 修复了 ncnn CLI 中 `--language English` 导致立即 EOS、输出为空的问题。
  原因是之前错误地把 `language English<asr_text>` 预填到了 assistant answer
  前缀里。现在不再这样预填。
- 增加了简单 fixed-window chunking，使 `pdx_voice` 这种 4.95 秒真实语音样本
  可以完整匹配 PyTorch。
- 增加了 chunk 文本拼接清理，避免中间 chunk 的句号导致结果变成
  `This is a test. Of me ...`。
- 通过重新导出 `text_seq_len=128` 解决了当前长文本样例在 text64 下被截断的问题。

## 9. 当前限制

- 每个音频 chunk 仍然是固定 256-frame mel window。
- `text_backbone` 已验证 `text_seq_len=64` 和 `text_seq_len=128`；更长输出需要
  根据目标场景继续增大静态长度或实现更完整的 cache/分段策略。
- 长音频只是简单切块和文本拼接，还不等价于官方 Python pipeline。
- 当前 decoding 是 greedy decode。
- KV cache 尚未实现。
- timestamp / forced aligner 尚未实现。
- 目前只完成 Linux/4090 VM 验证，还需要第二平台 build/smoke test。

## 10. 当前判断

核心结论：Qwen3-ASR 的主要神经网络模块可以通过 pnnx 转成 ncnn，并且当前
C++ runtime 已经可以在多个短英文语音样本上对齐官方 PyTorch 输出。

这说明转换路径是可行的。但当前分支仍应视为 prototype，而不是完整生产级 ASR
runtime。下一步应重点补齐：

- 更接近官方 pipeline 的长音频 chunk/overlap/stitching；
- 更大的静态 shape 或更灵活的导出策略；
- KV cache；
- 第二平台验证；
- PR 前的代码和文档整理。
