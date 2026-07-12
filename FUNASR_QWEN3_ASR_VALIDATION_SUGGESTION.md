# FunASR / Qwen3-ASR 兼容性验证建议

非认领这个任务，只从 FunASR / Qwen3-ASR 侧补一个验证建议，供后面做 ncnn
移植的同学参考。

这个任务里“ncnn 输出最终文本须与 PyTorch 原版一致”建议拆成几层测，否则很容易
只测到 demo 音频能跑。

## 1. 固定音频前处理

建议明确记录：

- sample rate；
- mono 转换方式；
- padding / truncation 策略；
- mel / filterbank 参数；
- chunk window 和 overlap 参数。

同时先保存 PyTorch 路径和 ncnn 路径的 encoder 输入摘要，例如：

- shape；
- dtype；
- min / max / mean / std；
- 前几个 frame 或前几个 bin 的统计量。

这样可以避免差异其实来自音频前处理，而不是模型转换或 ncnn 推理。

## 2. 端到端文本对齐

建议至少准备三类 fixture：

- 短中文普通话；
- 较长中文音频；
- 中文夹英文、数字、标点的音频。

比较时建议先比较最终 `normalized text`。如果模型输出带 timestamp、特殊 token
或额外标记，再单独比较清洗前文本和清洗后文本，避免把 postprocess 问题误判成
推理问题。

## 3. 模块级定位

Qwen3-ASR 这类结构通常可以按以下几段定位：

- speech encoder；
- projector / adaptor；
- Qwen decoder。

转换时最好分别记录：

- encoder embedding；
- projected embedding；
- 首轮 decoder logits。

并给出 PyTorch 与 ncnn 的误差范围，例如 `max_abs`、`mean_abs`、`p99_abs`。这样
排查 pnnx/ncnn 算子误差会快很多。

## 4. 多平台 smoke

Linux + Windows 编译运行之外，建议至少记录：

- CPU 型号和指令集；
- 线程数；
- RTF / 耗时；
- 峰值内存；
- 是否启用 Vulkan / GPU。

如果后续移动端走 Vulkan / ARM，建议单独补性能表。不要把移动端性能结论和
PyTorch accuracy parity 混在一个结论里。

## 5. 回归入口

最终技术总结里建议给出一个最小复测命令，例如：

```bash
qwen3-asr-ncnn \
  --model <model-dir> \
  --audio sample.wav \
  --text-out out.txt
```

这样 FunASR / Qwen3-ASR 用户可以直接复测，并反馈转换、前处理或后处理问题。

如果后续有公开 repo 或 PR，我可以从 FunASR / Qwen3-ASR 兼容性角度帮忙看一下
测试覆盖和输出契约。
