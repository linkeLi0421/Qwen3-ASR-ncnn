# Qwen3-ASR ncnn 中文技术记录

实现代码分支：

https://github.com/linkeLi0421/ncnn_llm/tree/qwen3-asr-export

当前实现提交：

```text
4eeda9f Improve Qwen3-ASR runtime decoding
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
- 已验证模型：`Qwen/Qwen3-ASR-0.6B`
- 当前导出：`fp32`
- 当前静态 text 长度：`text_seq_len=64`
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

| 样本 | 时长 | PyTorch | ncnn | 是否一致 |
| --- | ---: | --- | --- | --- |
| `pdx_voice` | 4.95s | `This is a test of me recording my voice.` | `This is a test of me recording my voice.` | 是 |
| `hello_world` | 1.36s | `Hello world.` | `Hello world.` | 是 |
| `natural_zero` | 0.64s | `Zero.` | `Zero.` | 是 |
| `digit_five` | 0.42s | `Five.` | `Five.` | 是 |
| `long_digit_five` | 16.97s | `By by` | `By by by by by by by by by by ...` | 否 |

`long_digit_five` 是人工重复样本，不适合作为语义正确性的主要 benchmark。
它的失败应记录为长音频 chunk stitching 限制，不应视为核心 ncnn 转换失败。

## 7. 当前限制

- 长音频 chunking 仍是简单固定窗口。
- 每个 chunk 独立 decode，再做简单文本拼接。
- `text_seq_len=64` 限制 prompt + 生成文本长度。
- 还没有 KV cache。
- 还没有 timestamps / forced aligner。
- 还没有第二平台验证。

## 8. 下一步

1. 做第二平台 build/smoke test。
2. 改进长音频 overlap chunk 和拼接。
3. 增加 KV cache。
4. 做更大静态 shape 或更灵活的导出策略。
5. 整理 PR 说明和 Discussion 回复。
