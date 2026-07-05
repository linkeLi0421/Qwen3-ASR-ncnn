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
- 还没有 KV cache。
- 还没有 timestamps / forced aligner。
- 还没有第二平台验证。

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

结论：长文本截断不是 ncnn runtime 逻辑错误，而是 `text_backbone` 静态长度太小。
通过 `--text-seq-len 128` 重新导出可以解决当前样例。

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

1. 做第二平台 build/smoke test。
2. 继续改进长音频 overlap/stitching，例如加入更稳的 overlap 长度选择、
   置信度/时间戳辅助和跨 chunk 上下文。
3. 增加 KV cache。
4. 做更大静态 shape 或更灵活的导出策略。
5. 整理 PR 说明和 Discussion 回复。
