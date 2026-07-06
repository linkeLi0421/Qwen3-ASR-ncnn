# Tencent/ncnn discussion 6805 更新

我更新了 Qwen3-ASR 转 ncnn 的原型实现和实验记录。

实现分支：

- https://github.com/linkeLi0421/ncnn_llm/tree/qwen3-asr-export

实验记录和测试样例：

- https://github.com/linkeLi0421/Qwen3-ASR-ncnn

当前实现分支最新提交：

```text
89394d2 Gate Qwen3-ASR timing logs behind flag
```

## 当前状态

目前这个原型可以把 `Qwen/Qwen3-ASR-0.6B` 拆成几个可以通过 pnnx 转成 ncnn
的模块：

- `audio_encoder`
- `text_embed`
- `text_backbone`
- `lm_head`

C++ runtime 目前已经包含：

- WAV 读取和 log-mel 特征提取
- prompt/audio token 组装
- audio embedding 替换
- greedy decoding
- 固定 256-frame mel window 的 chunking
- 简单的 overlap/stitching 长音频拼接
- 实验性的 KV cache decode，运行参数是 `--use-kv-cache`
- 可选 timing 输出，运行参数是 `--timing`

## 平台 smoke test

同一个 `pdx_voice_16k.wav` 样例已经在两个平台上验证：

| 平台 | 结果 |
| --- | --- |
| Linux VM，机器有 RTX 4090，当前使用 CPU ncnn runtime | `This is a test of me recording my voice.` |
| macOS 15.7.4，Apple M1 Pro，arm64，CPU-only | `This is a test of me recording my voice.` |

## 长文本和 KV cache 结果

之前静态导出的 `text_seq_len=64` 对更长输出会截断。重新导出更大的静态 shape
并加入实验性 KV cache 路径后，结果如下：

| Runtime | 结果 |
| --- | --- |
| static text64 | 在 `fourteen` 前截断 |
| static text128 | 可以输出完整句子 |
| KV48 | 可以输出完整句子，并且和 static text128 一致 |

在更长的 one-to-thirty 合成样例上，当前测试环境里 KV48 和 static text128
输出也一致。

## KV cache timing

我给 `qwen3_asr_main` 加了 `--timing` 参数，避免默认运行时打印大量 timing
日志。在 Linux VM 上使用 CPU ncnn runtime，参数为 `--threads 8 --timing`，
KV48 相比 static text128 有明显加速：

| 样例 | Static total | KV total | Total speedup | Decode speedup |
| --- | ---: | ---: | ---: | ---: |
| `pdx_voice` | 16043.5 ms | 2578.2 ms | 6.22x | 6.73x |
| `long_text_numbers_fast` | 13572.5 ms | 2677.9 ms | 5.07x | 5.61x |
| `long_text_numbers_30` | 47290.9 ms | 13902.6 ms | 3.40x | 3.57x |

这里需要特别说明：这个 timing 是 CPU ncnn runtime 的结果，不是 RTX 4090 GPU
timing。

## 当前限制

- KV cache 仍然是实验性实现。
- 当前 KV48 可以跑通，但 KV64/KV128 的导出矩阵还需要继续排查。
- 长音频 overlap/stitching 只是一个简单 baseline，还不等价于官方 Python
  pipeline。
- 还没有实现 timestamps / forced alignment。
- 当前 4090 VM 上 `--vulkan` 只枚举到 `llvmpipe` 软件 Vulkan device，没有枚举到
  RTX 4090；Vulkan 输出也不可靠，所以目前不声明 GPU 加速结果。

目前的结论是：Qwen3-ASR 的核心 ncnn 转换路径已经跑通；短音频端到端输出可以
对齐 PyTorch；实验性 KV cache 路径在 CPU ncnn runtime 上有明确 decode
加速。下一步主要是继续改进长音频拼接，排查 KV64/KV128 导出问题，以及单独解决
Vulkan/RTX 4090 设备可见性问题。
