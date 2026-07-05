# 失败与重试记录

这个文件记录实验过程中遇到的主要失败、误判和修复过程，方便后续讨论时说明
当前结果是怎样一步步得到的。

## 1. 不能直接转换完整 Hugging Face pipeline

最开始的直觉是把完整 Qwen3-ASR pipeline 直接交给 pnnx。这个方向不可行，
因为官方 `transcribe()` 不只是模型 forward，还包含大量 Python 控制逻辑：

- 音频加载与重采样；
- processor / feature extractor；
- 动态音频 chunk；
- 特殊 audio token；
- prompt 构造；
- generation wrapper；
- 输出解析。

重试方案：把模型拆成 `audio_encoder`、`text_embed`、`text_backbone`、`lm_head`
四个静态模块，把 Python pipeline 逻辑迁移到 C++ runtime。

## 2. bf16 转换不稳定

bf16 TorchScript 可用于检查模型结构，但 pnnx/ncnn runtime 验证时不够稳定。

重试方案：使用 `fp32` 作为当前可靠验证路径。当前 `--convert-ncnn` 要求
`--dtype fp32`。

## 3. audio encoder 有动态 Python 逻辑

官方 audio encoder forward 里有动态切块、padding、mask、ragged sequence 等逻辑，
不适合直接作为一个通用动态图转换。

重试方案：实现 batch=1 的静态 audio wrapper，用固定 `[128, 256]` mel 输入验证。
该 wrapper 已和原始 PyTorch audio encoder 对齐：

```text
max_abs 3.7625e-06
mean_abs 2.4569e-07
```

## 4. C++ log-mel 前处理需要校准

如果 C++ 前处理和 Hugging Face processor 不一致，即使 ncnn 模型本身正确，
最终文本也会偏。

重试方案：实现 Whisper/Qwen3-ASR 风格 log-mel，并和 Hugging Face 路径对比。
当前误差：

```text
max_abs 1.704692840576172e-05
mean_abs 1.3335738913156092e-07
p99_abs 2.384185791015625e-07
```

## 5. 误判：ncnn 没有输出文本

一开始测试脚本中 ncnn 只打印了 `audio_encoder` shape，没有打印 `text=`。
这不是模型坏了，而是脚本没有传 `--generate-from-features`，导致只跑了音频
encoder，没有进入 decode。

修复：测试脚本和手工命令都加上：

```bash
--generate-from-features
```

## 6. `--language English` 导致空输出

早期 C++ prompt 处理里把 `language English<asr_text>` 直接预填到 assistant
回答前缀中。这样再 decode 时，模型可能立刻输出 EOS，最终 `text=` 为空。

修复：不再把 forced language 文本预填进 assistant response。现在
`--language English` 不会导致 `hello_world` 空输出。

验证：

```text
text=Hello world.
```

## 7. `pdx_voice` 一开始只识别到前半句

`pdx_voice` 是约 4.95 秒的真实语音：

```text
This is a test of me recording my voice.
```

早期 ncnn 输出：

```text
This is a test.
```

原因：runtime 只处理第一个固定 256-frame mel window，没有覆盖完整音频。

修复：为 WAV 输入增加简单 fixed-window chunking，并把多个 chunk 的文本拼接。

修复后结果：

```text
PyTorch: This is a test of me recording my voice.
ncnn:    This is a test of me recording my voice.
```

## 8. chunk 拼接出现大小写/标点问题

加入 chunking 后，中间结果曾出现：

```text
This is a test. Of me recording my voice.
```

原因：每个 chunk 独立 decode，中间 chunk 可能自己生成句号，下一个 chunk 又以
大写开头。

修复：对非最后 chunk 去掉结尾的 `. ! ?`，拼接时把后续 chunk 开头的大写英文
字母转小写。

## 9. `long_digit_five` 仍不匹配

`long_digit_five` 是把同一个 `Five.` 短音频重复 40 次得到的人工压力样本。
它不是自然语音，也不是很好的 ASR 语义 benchmark。

结果：

```text
PyTorch: By by
ncnn:    By by by by by by by by by by ...
```

分析：PyTorch 自己也没有输出重复的 `Five.`，说明这个样本本身会使模型困惑。
ncnn 当前按固定窗口逐块 decode，重复音频会自然导致重复文本。这个失败应该记录
为长音频 chunk stitching 限制，而不是 ncnn 模块转换失败。

后续如果要真正解决，需要：

- overlap chunk；
- timestamp 或置信度辅助拼接；
- 去重策略；
- 更接近官方 pipeline 的音频分段逻辑。

## 10. VM 上的 ncnn CPU 信息问题

验证 VM 上曾遇到 `ncnn::get_cpu_count()` 崩溃，原因和 CPU cache parsing 中
shared physical CPU count 为 0 有关。

处理：VM 上本地 patch 了 ncnn 源码以继续验证。这个 patch 还没有并入
`ncnn_llm`，需要后续单独处理或向 ncnn 上游反馈。
