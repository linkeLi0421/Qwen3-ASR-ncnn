# 当前状态

## 结论

当前实现证明了 Qwen3-ASR 转 ncnn 的技术路径可行，但还没有达到“ncnn 输出最终文本
与 PyTorch 原版一致”的完成标准。

已经完成：

- 将 Qwen3-ASR 拆成可转换的 ncnn 模块。
- C++ runtime 已支持 16 kHz mono PCM16 WAV、log-mel、prompt/audio token 组装、
  audio embedding 替换、greedy decode。
- 本机新增三类中文 fixture 分层测评。
- ncnn 侧可输出 mel summary、最终 JSON、首轮 decoder summary。
- macOS CPU-only 已记录 RTF 和峰值内存。
- Linux VM 已跑通原版 PyTorch baseline，包含 text 和 mel summary。
- Linux VM 已跑通 ncnn CPU smoke，并记录 RTF 和峰值内存。
- C++ runtime 已修正 forced language prompt，`--language Chinese` 路径与 PyTorch
  prompt 对齐。
- PyTorch fixed-chunk 模块 summary 已补，覆盖 audio embedding、merged embedding、
  hidden、logits 和 selected logits。
- 已定位长音频失败根因：256-frame 独立 chunk 路径和官方整段 `transcribe()` 不等价。
- 已验证 long-window ncnn 路径：`audio_frames=1878`、`text_seq_len=512`、KV cache
  可以让 `zh_long_tts` normalized text 对齐 PyTorch。

未完成：

- PyTorch 与 ncnn 的模块级数值误差表。
- long-window KV 模型的三类 fixture 完整复测。
- 真实中文录音 fixture。
- Windows 最新分层 smoke。
- 4090/Vulkan 有效性能验证。

## 完成标准

| 层级 | 当前状态 | 还缺什么 |
| --- | --- | --- |
| 音频前处理 parity | ncnn C++ mel summary 和 PyTorch mel summary 已有 | 数值 diff |
| 三类中文 fixture | 本机 TTS fixture 已有，PyTorch baseline 已有 | 真人中文录音 |
| normalized text 对齐 | text128 旧基线：ncnn strict 1/3，semantic 2/3；long-window KV：`zh_long_tts` 通过 | long-window 三类 fixture 完整复测 |
| 模块级定位 | ncnn 侧 summary 已有；PyTorch fixed-chunk summary 已有 | encoder/projector/decoder 首轮 logits 数值误差表 |
| 多平台 smoke | macOS CPU-only、Linux CPU 已有 | Windows smoke，4090/Vulkan |
| 最小复测命令 | 已支持 | 在最终报告里固定命令和输出契约 |

## 当前最重要的判断

现在不应该继续用旧的英文 demo 或零散压力样例作为完成依据。它们只能作为历史
smoke 或调试材料。当前主线是 `tests/qwen3_asr` 分层测评。

## Linux VM 补充

Linux VM 上 PyTorch baseline 显示：短中文和长中文与 expected 一致；混合中英数字样例
PyTorch 输出 `Open API`，而 fixture expected 是 `OpenAI API`，因此这个样例的 strict
失败不能单独归因给 ncnn。

ncnn Linux CPU smoke 需要注意一个环境问题：原始 ncnn build 在该 VM 上会在
`get_data_cache_size()` 触发 `SIGFPE`。原因是 CPU cache `shared_cpu_map` 形如
`00000000,ffffffff,00000000,ffffffff`，当前解析路径可能得到 0 个 physical core 并除零。
本次 Linux smoke 使用本地临时 patched ncnn build 继续验证 Qwen3-ASR runtime；这不是
4090/Vulkan 结果。

最新 forced prompt 结果：短中文 strict/semantic 均通过，长中文仍 strict 失败；
混合中英数字 ncnn semantic 通过但 strict 失败。Linux CPU RTF 分别约为 4.54、3.60、
3.72，峰值 RSS 约 4.74 GiB。

最新 long-window KV 结果：18.78 秒 `zh_long_tts` 不再拆成 9 个 256-frame chunks，
而是按整段 `audio_frames=1878` 输入。ncnn audio encoder 输出 244 个 audio tokens，
prompt_len 为 262；`text_seq_len=512` 足够容纳 prompt 和输出。CPU 路径完整生成
用时 70.29 秒，峰值 RSS 约 19.31 GiB，输出：

```text
这是一段较长的中文语音测试。我们希望模型能够稳定地识别连续的句子，并且在音频变长以后仍然保持合理的文本输出。这个样例主要用于检查切块拼接以及端到端文本对齐是否可靠。
```

这说明旧错误不是 ncnn 算子把 `检查/对齐` 算错，而是短窗独立 ASR 造成的 pipeline
不等价。代价是模型包和内存显著增大，且当前 Vulkan 仍只看到 `llvmpipe`。
