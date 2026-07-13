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

未完成：

- PyTorch 同 schema summary。
- PyTorch 与 ncnn 的 normalized text 对齐表。
- PyTorch 与 ncnn 的模块级误差表。
- 真实中文录音 fixture。
- Linux/Windows 最新分层 smoke。
- 4090/Vulkan 有效性能验证。

## 完成标准

| 层级 | 当前状态 | 还缺什么 |
| --- | --- | --- |
| 音频前处理 parity | ncnn C++ mel summary 已有 | PyTorch mel summary 和数值 diff |
| 三类中文 fixture | 本机 TTS fixture 已有 | 真人中文录音和 PyTorch baseline |
| normalized text 对齐 | strict 1/3，semantic 2/3 | PyTorch 对齐后再判断是转换问题还是模型/后处理问题 |
| 模块级定位 | ncnn 侧 summary 已有 | PyTorch encoder/projector/decoder 首轮 logits summary |
| 多平台 smoke | macOS CPU-only 已有 | Linux/Windows 最新分层 smoke |
| 最小复测命令 | 已支持 | 在最终报告里固定命令和输出契约 |

## 当前最重要的判断

现在不应该继续用旧的英文 demo 或零散压力样例作为完成依据。它们只能作为历史
smoke 或调试材料。当前主线是 `tests/qwen3_asr` 分层测评。
