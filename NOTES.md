# Qwen3-ASR ncnn 中文技术记录

实现代码分支：

https://github.com/linkeLi0421/ncnn_llm/tree/qwen3-asr-export

当前实现提交：

```text
5f1fab1 Add overlap stitching for Qwen3-ASR chunks
```

## 1. 转换目标

目标是把 `Qwen/Qwen3-ASR-0.6B` 从 Hugging Face/PyTorch checkpoint 转成 ncnn
可运行的 `.ncnn.param` / `.ncnn.bin`，并通过 C++ runtime 完成 ASR 推理。

ncnn 不能直接运行 Hugging Face checkpoint。pnnx 需要的是静态 PyTorch /
TorchScript 计算图，而 Qwen3-ASR 的官方接口包含大量 Python pipeline 逻辑。

## 2. 为什么要拆模型

官方 `Qwen3ASRModel.transcribe()` 实际包含：

1. 读取音频；
2. 重采样和归一化；
3. 生成 log-mel 特征；
4. 音频切块和 padding；
5. audio encoder；
6. 构造带 audio placeholder token 的 prompt；
7. tokenizer；
8. text embedding；
9. 用 audio embedding 替换 audio token 位置；
10. text backbone autoregressive decode；
11. lm head；
12. detokenize；
13. 解析 language/text/timestamps。

其中只有一部分是纯神经网络静态图。当前转换拆成：

- `audio_encoder`
- `text_embed`
- `text_backbone`
- `lm_head`

C++ runtime 负责复现外围 pipeline。

## 3. 当前状态

- 已实现 exporter：`export/qwen3_asr_export.py`
- 已实现 C++ runtime：`qwen3_asr_main`
- 已加入正式 CMake 构建：`qwen3_asr_main`
- 已验证模型：`Qwen/Qwen3-ASR-0.6B`
- 当前导出：`fp32`
- 当前已验证静态 text 长度：`text_seq_len=64` 和 `text_seq_len=128`
- 当前每个音频 chunk：256-frame mel window
- 已支持 16 kHz mono PCM16 WAV 输入

## 4. 环境

- GPU：NVIDIA RTX 4090 24 GB
- Driver：`570.124.06`
- CUDA：`12.8`
- pnnx / ncnn：在验证机器本地编译
- PyTorch：CUDA 版本，用于 exporter 和官方结果对比

## 5. 模块级验证

| 模块 | 对比结果 |
| --- | --- |
| `text_embed` | 完全一致，`max_abs 0.0`，`mean_abs 0.0` |
| `lm_head` | `max_abs 6.6757e-06`，`mean_abs 3.6068e-07` |
| `text_backbone` | `max_abs 0.026946`，`mean_abs 0.0030005`，`p99_abs 0.0120745` |
| `audio_encoder` | `max_abs 2.5416e-05`，`mean_abs 1.2091e-06`，`p99_abs 8.7405e-06` |

C++ log-mel 与 Hugging Face 路径对比：

| 指标 | 数值 |
| --- | ---: |
| `max_abs` | `1.704692840576172e-05` |
| `mean_abs` | `1.3335738913156092e-07` |
| `p99_abs` | `2.384185791015625e-07` |

## 6. 端到端结果

| 样本 | runtime | 时长 | PyTorch | ncnn | 结论 |
| --- | --- | ---: | --- | --- | --- |
| `pdx_voice` | text128 + CMake | 4.95s | `This is a test of me recording my voice.` | `This is a test of me recording my voice.` | 通过 |
| `hello_world` | text64 | 1.36s | `Hello world.` | `Hello world.` | 通过 |
| `natural_zero` | text64 | 0.64s | `Zero.` | `Zero.` | 通过 |
| `digit_five` | text64 | 0.42s | `Five.` | `Five.` | 通过 |
| `long_text_numbers_fast` | text64 | 2.22s | `One two three four five six seven eight nine ten eleven twelve thirteen fourteen.` | `One two three four five six seven eight nine ten eleven twelve thirteen` | 截断 |
| `long_text_numbers_fast` | text128 | 2.22s | `One two three four five six seven eight nine ten eleven twelve thirteen fourteen.` | `One two three four five six seven eight nine ten eleven twelve thirteen fourteen.` | 通过 |
| `long_text_numbers_fast` | text48 + KV cache | 2.22s | `One two three four five six seven eight nine ten eleven twelve thirteen fourteen.` | `One two three four five six seven eight nine ten eleven twelve thirteen fourteen.` | 通过 |
| `long_text_numbers_30` | text64 | 7.00s | 合成 one 到 thirty | `One two ... thirteen 16, 17, 18, 1 twenty-four ... twenty` | 截断/拼接不完整 |
| `long_text_numbers_30` | text128 | 7.00s | 合成 one 到 thirty | `One two ... sixteen 16, 17, 18, 19, 20, 21, 22, 23 twenty-four ... thirty.` | 通过当前拼接基线 |
| `long_text_numbers_30` | text48 + KV cache | 7.00s | 合成 one 到 thirty | 与 text128 输出一致 | 通过当前拼接基线 |
| `long_digit_five` | text128 overlap chunking | 16.97s | `By by` | `By by by by by by five, five, five.` | 长音频拼接限制 |

`long_digit_five` 是人工重复样本，不适合作为语义正确性的主要 benchmark。
它的失败应记录为长音频 chunk stitching 限制，不应视为核心 ncnn 转换失败。
加入 32-frame overlap 和词级去重后，输出从早期的长重复/混合片段改善为
`By by by by by by five, five, five.`，但仍不应作为语义正确性通过样本。

## 7. 当前限制

- 长音频 chunking 已加入 32-frame overlap 和词级去重拼接，但仍然不是官方 Python pipeline 的完整复刻。
- 每个 chunk 仍然独立 decode，再做文本拼接。
- `text_seq_len=64` 会限制 prompt + 生成文本长度；已验证重新导出
  `text_seq_len=128` 可以解决当前长文本样例截断问题。
- 已加入实验性 KV cache 导出/runtime 路径。当前在 6 条短音频样例和 2 条长输出
  压力样例上与对应静态路径对齐。KV48 已验证；KV64 重新导出在当前 VM 上卡在
  模型加载/初始化阶段，需要后续单独排查。
- 还没有 timestamps / forced aligner。
- 已完成第二平台 macOS arm64 CPU-only smoke test。

## 8. CMake 验证

正式 CMake 已在 Linux/RTX 4090 VM 上通过：

```bash
cmake -S /data/qwen3-asr-ncnn/src/ncnn_llm \
  -B /data/qwen3-asr-ncnn/build/ncnn_llm_cmake \
  -Dncnn_DIR=/data/qwen3-asr-ncnn/build/ncnn/install/lib/cmake/ncnn \
  -DNCNN_LLM_NLOHMANN_JSON_INCLUDE_DIR=/usr/local/lib/python3.12/dist-packages/include/cudnn_frontend/thirdparty

cmake --build /data/qwen3-asr-ncnn/build/ncnn_llm_cmake \
  --target qwen3_asr_main -j
```

构建结果：

```text
[100%] Built target qwen3_asr_main
```

使用正式 CMake binary 验证 `pdx-cs-sound/wavs/voice.wav` 转换后的
`pdx_voice_16k.wav`：

```text
text=This is a test of me recording my voice.
```

第二平台 macOS arm64 CPU-only 也已通过：

| 项目 | 值 |
| --- | --- |
| 系统 | macOS 15.7.4 |
| 芯片 | Apple M1 Pro |
| 架构 | arm64 |
| ncnn | 本机源码 CPU-only build，`NCNN_VULKAN=OFF` |
| ncnn_llm | 本机 CMake build |
| binary | `/Users/link/llk/build/ncnn_llm-macos-qwen3-asr/qwen3_asr_main` |
| 模型 | `/Users/link/llk/models/qwen3_asr_0_6b_runtime_text128` |
| 音频 | `/Users/link/llk/test_audio/pdx_voice_16k.wav` |

本机 build 命令：

```bash
cmake -S /Users/link/llk/ncnn \
  -B /Users/link/llk/build/ncnn-macos-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/Users/link/llk/build/ncnn-macos-install \
  -DNCNN_VULKAN=OFF \
  -DNCNN_BUILD_TOOLS=OFF \
  -DNCNN_BUILD_EXAMPLES=OFF \
  -DNCNN_BUILD_TESTS=OFF \
  -DNCNN_BUILD_BENCHMARK=OFF \
  -DNCNN_SHARED_LIB=OFF

cmake --build /Users/link/llk/build/ncnn-macos-cpu -j$(sysctl -n hw.ncpu)
cmake --install /Users/link/llk/build/ncnn-macos-cpu

cmake -S /Users/link/llk/ncnn_llm \
  -B /Users/link/llk/build/ncnn_llm-macos-qwen3-asr \
  -DCMAKE_BUILD_TYPE=Release \
  -Dncnn_DIR=/Users/link/llk/build/ncnn-macos-install/lib/cmake/ncnn \
  -DNCNN_LLM_NLOHMANN_JSON_INCLUDE_DIR=/opt/homebrew/include

cmake --build /Users/link/llk/build/ncnn_llm-macos-qwen3-asr \
  --target qwen3_asr_main -j$(sysctl -n hw.ncpu)
```

本机 smoke test 命令：

```bash
/Users/link/llk/build/ncnn_llm-macos-qwen3-asr/qwen3_asr_main \
  --model /Users/link/llk/models/qwen3_asr_0_6b_runtime_text128 \
  --audio-wav /Users/link/llk/test_audio/pdx_voice_16k.wav \
  --generate-from-features --max-new-tokens 32 --threads 6
```

输出：

```text
text=This is a test of me recording my voice.
```

## 9. 长文本验证

为验证 `text_seq_len=64` 的限制，生成了一个 2.22 秒快速数字语音：

```bash
espeak-ng -s 340 -w long_text_numbers_fast.wav \
  "one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
ffmpeg -y -i long_text_numbers_fast.wav \
  -ar 16000 -ac 1 long_text_numbers_fast_16k.wav
```

PyTorch 输出：

```text
One two three four five six seven eight nine ten eleven twelve thirteen fourteen.
```

text64 ncnn 输出被静态长度截断：

```text
One two three four five six seven eight nine ten eleven twelve thirteen
```

重新导出并组装 `qwen3_asr_0_6b_runtime_text128` 后，ncnn 输出：

```text
One two three four five six seven eight nine ten eleven twelve thirteen fourteen.
```

同一样例在 KV cache 路径下使用 `qwen3_asr_0_6b_runtime_kv48` 也输出完整句子。
这里的原因是 KV 路径只要求 prefill 的 prompt/audio 长度不超过 `text_seq_len`；
后续生成 token 通过 cache 增长，不再像静态 decode 一样每步受完整
`prompt + generated` 长度限制。

结论：静态 decode 的长文本截断不是 ncnn runtime 逻辑错误，而是 `text_backbone`
静态长度太小。通过 `--text-seq-len 128` 重新导出可以解决当前样例；KV cache
路径可以绕开“每步重跑完整序列”的长度增长问题。

进一步生成了 7.00 秒的 one 到 thirty 合成长输出样例：

```bash
espeak-ng -s 360 -w long_text_numbers_30_raw.wav \
  "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty twenty one twenty two twenty three twenty four twenty five twenty six twenty seven twenty eight twenty nine thirty"
ffmpeg -y -i long_text_numbers_30_raw.wav \
  -ar 16000 -ac 1 long_text_numbers_30_16k.wav
```

结果：

| runtime | chunk 数 | 生成 token 数 | 输出摘要 | 结论 |
| --- | ---: | ---: | --- | --- |
| text64 | 3 | 48 | `One two ... thirteen 16, 17, 18, 1 twenty-four ... twenty` | 截断/拼接不完整 |
| text128 | 3 | 77 | `One two ... sixteen 16, 17, 18, 19, 20, 21, 22, 23 twenty-four ... thirty.` | 通过当前拼接基线 |
| text48 + KV cache | 3 | 77 | 与 text128 完全一致 | 通过当前拼接基线 |

这个样例仍经过 chunking，因此它同时测试了“较长输出”和“跨 chunk 拼接”。
text128 与 KV48 对齐，说明 KV cache 的多步生成在当前长输出压力下没有漂移。

## 10. 长音频 overlap/stitching 验证

runtime 已加入带 overlap 的固定窗口 chunking：

```bash
--chunk-overlap-frames 32
```

拼接时会对相邻 chunk 的文本做词级 suffix/prefix 去重。回归结果：

| 样本 | 结果 |
| --- | --- |
| `pdx_voice_16k.wav` | 仍输出 `This is a test of me recording my voice.` |
| `long_text_numbers_fast_16k.wav` | 仍输出完整数字序列 |
| `long_digit_5_x40_16k.wav` | 输出 `By by by by by by five, five, five.`，仍记录为人工重复压力样例失败 |

## 11. 下一步

1. 排查 KV64 重新导出在当前 VM 上卡在模型加载/初始化阶段的问题。
2. 排查 4090 VM 上 ncnn Vulkan 只看到 `llvmpipe` 软件设备的问题；当前 timing
   仍是 CPU ncnn runtime，不是 RTX 4090 GPU timing。
3. 继续改进长音频 overlap/stitching，例如加入更稳的 overlap 长度选择、
   置信度/时间戳辅助和跨 chunk 上下文。
4. 做更大静态 shape 或更灵活的导出策略。
5. 整理 PR 说明和 Discussion 回复。

## 12. KV cache 实验记录

目标：让文本 decoder 不必每生成一个 token 都重新跑完整 prompt，而是先跑一次
prefill 得到每层 K/V cache，再逐 token decode。

当前实现：

- exporter 新增 `--export-kv-cache` 和 `--kv-cache-len`。
- 额外导出 `text_prefill_kv` 和 `text_decode_kv`。
- C++ runtime 新增 `--use-kv-cache`，默认仍走稳定的静态 decode。
- decode 图把 RoPE 的 `cos/sin` 作为输入，由 C++ 按当前 position 生成。

关键问题和修复：

1. 直接用 Transformers `use_cache=True` trace 会在 mask/cache 逻辑中失败，因此改成
   手写每层 attention 的 prefill/decode wrapper。
2. 初版 one-token decode 图按 `cache_len=48` trace，pnnx 把 cache concat 后的
   repeated-KV reshape 写死成 `1=49`。第一步 decode 正常，第二步 cache 变成 49
   后再 concat 到 50，但图仍按 49 reshape，输出开始漂移。
3. 尝试在 C++ 中把 cache 裁回固定窗口会破坏 prompt/audio context，结果更差。
4. 使用 `inputshape2` 可以让 pnnx 推断动态 cache 维度，但生成的 ncnn param 带
   `pnnx.Expression`，当前 runtime 不能加载。
5. 最终采用后处理：只把 `text_decode_kv.ncnn.param` 中 cache repeat 后的
   `Reshape 0=128 1=49 2=16` 改成 `Reshape 0=128 1=-1 2=16`。ncnn 可以加载，
   并允许 cache 随 decode 步数增长。

验证环境：

- 默认静态路径：`qwen3_asr_0_6b_runtime_text128`
- KV cache 路径：`qwen3_asr_0_6b_runtime_kv48`
- binary：正式 CMake build 的 `qwen3_asr_main`
- 平台：Linux VM + RTX 4090

短样例批量对比结果：

| 样本 | 时长 | 默认静态 decode | KV cache decode | 结论 |
| --- | ---: | --- | --- | --- |
| `pdx_voice_16k.wav` | 4.95s | `This is a test of me recording my voice.` | `This is a test of me recording my voice.` | 对齐 |
| `pdx_voice-note_16k.wav` | 1.40s | `啊。` | `啊。` | 对齐 |
| `0_jackson_0_16k.wav` | 0.64s | `Zero.` | `Zero.` | 对齐 |
| `1_jackson_0_16k.wav` | 0.52s | `一。` | `一。` | 对齐 |
| `natural_digit_0_16k.wav` | 0.64s | `Zero.` | `Zero.` | 对齐 |
| `random_digit_5_jackson_0_16k.wav` | 0.42s | `Five.` | `Five.` | 对齐 |

补充：原始 `0_jackson_0.wav` 和 `1_jackson_0.wav` 是 8 kHz，runtime 会按预期拒绝；
测试时先用 `ffmpeg -ar 16000 -ac 1` 转成 16 kHz。

长输出对比结果：

| 样本 | 默认静态 decode | KV cache decode | 结论 |
| --- | --- | --- | --- |
| `long_text_numbers_fast_16k.wav` | text128 输出完整 one 到 fourteen | KV48 输出一致 | 对齐 |
| `long_text_numbers_30_16k.wav` | text128 生成 77 tokens | KV48 生成 77 tokens，输出一致 | 对齐 |

不同 `text_seq_len` 结论：

- 静态 text64 会因为 `prompt + generated` 长度耗尽而截断。
- 静态 text128 可以覆盖当前长文本样例。
- KV48 的 prefill 长度刚好等于当前 prompt 长度 48，后续生成靠 cache 增长；
  因此它可以输出超过 48 token context 的长结果。
- KV64 重新导出尝试过一次，但当前 VM 上进程长时间停在模型加载/初始化阶段，
  `/tmp/qwen3_kv64_export.log` 只有 `torch_dtype` deprecation warning，输出目录为空；
  这不是 ncnn runtime 失败，需要后续单独排查导出环境/processor 加载。

结论：KV cache 路径在短样例和当前长输出样例上已经和默认静态路径对齐。它仍应
标记为实验性，因为还没有完成 KV64/KV128 导出矩阵，也还需要更多平台/样例覆盖。

## 13. KV cache timing

为判断 KV cache 是否只有结构意义，还是已经带来实际性能收益，给
`qwen3_asr_main` 增加了粗粒度 timing 输出：

- `audio_encoder_time_ms`
- `prefill_time_ms`
- `decode_step_time_ms[i]`
- `decode_total_time_ms`

测试环境：

| 项目 | 值 |
| --- | --- |
| 平台 | Linux VM |
| GPU | RTX 4090 24 GB present |
| ncnn runtime | CPU path，`--vulkan` 未启用 |
| threads | 8 |
| static model | `qwen3_asr_0_6b_runtime_text128` |
| KV model | `qwen3_asr_0_6b_runtime_kv48` |

结果：

| 样本 | 模式 | chunk | tokens | audio ms | decode ms | total measured ms | 输出 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `pdx_voice` | static text128 | 2 | 17 | 404.098 | 15639.420 | 16043.518 | 正确 |
| `pdx_voice` | KV48 | 2 | 17 | 254.509 | 2323.650 | 2578.159 | 正确 |
| `long_text_numbers_fast` | static text128 | 1 | 18 | 143.417 | 13429.100 | 13572.517 | 正确 |
| `long_text_numbers_fast` | KV48 | 1 | 18 | 284.830 | 2393.070 | 2677.900 | 正确 |
| `long_text_numbers_30` | static text128 | 3 | 77 | 642.555 | 46648.300 | 47290.855 | 对齐当前基线 |
| `long_text_numbers_30` | KV48 | 3 | 77 | 842.902 | 13059.710 | 13902.612 | 对齐当前基线 |

速度比：

| 样本 | total measured speedup | decode-only speedup | per-step avg speedup |
| --- | ---: | ---: | ---: |
| `pdx_voice` | 6.22x | 6.73x | 7.46x |
| `long_text_numbers_fast` | 5.07x | 5.61x | 6.02x |
| `long_text_numbers_30` | 3.40x | 3.57x | 4.01x |

解释：

- KV cache 在当前 CPU runtime 上已经有明显收益，尤其是 token-by-token text
  decoder 部分。
- audio encoder 时间不稳定，且不是 KV cache 优化目标；判断 KV 收益应主要看
  decode-only 和 per-step avg。
- 长输出样例里 speedup 降低到约 3.4x，是因为多 chunk、prefill 和 cache I/O
  占比上升。
- 这还不是 RTX 4090 GPU timing。检查 `--vulkan` 时，ncnn 只枚举到
  `llvmpipe (LLVM 20.1.2, 256 bits)` 软件 Vulkan device，而不是 4090；
  并且 `--vulkan` smoke 输出不可靠，因此当前不把 Vulkan 结果作为有效性能数据。
