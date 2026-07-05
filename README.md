# Qwen3-ASR ncnn 转换记录

这个仓库用于记录 `Qwen/Qwen3-ASR-0.6B` 转 ncnn 的实验过程、当前结果、
失败原因和后续计划。

实际代码实现不在本仓库，代码分支在：

https://github.com/linkeLi0421/ncnn_llm/tree/qwen3-asr-export

当前实现分支最新提交：

```text
4eeda9f Improve Qwen3-ASR runtime decoding
```

## 当前结论

目前已经验证：`Qwen/Qwen3-ASR-0.6B` 的核心神经网络模块可以通过
TorchScript + pnnx 转成 ncnn，并且 C++ ncnn runtime 可以在多个短音频样本上
和官方 PyTorch `Qwen3ASRModel.transcribe()` 输出一致。

已匹配的样本包括：

| 样本 | 时长 | PyTorch | ncnn | 是否一致 |
| --- | ---: | --- | --- | --- |
| `pdx_voice` | 4.95s | `This is a test of me recording my voice.` | `This is a test of me recording my voice.` | 是 |
| `hello_world` | 1.36s | `Hello world.` | `Hello world.` | 是 |
| `natural_zero` | 0.64s | `Zero.` | `Zero.` | 是 |
| `digit_five` | 0.42s | `Five.` | `Five.` | 是 |

仍未完全解决的是长音频处理。当前长音频只是简单按固定窗口切块后拼接文本，
还不等价于官方 Python pipeline。

## 仓库文件

- [NOTES.md](./NOTES.md)：完整中文技术记录。
- [DISCUSSION_REPORT.md](./DISCUSSION_REPORT.md)：可复制到 GitHub Discussion 的中文报告。
- [FAILURE_RETRY_LOG.md](./FAILURE_RETRY_LOG.md)：失败、误判、重试和修复记录。
- [TESTCASES.md](./TESTCASES.md)：测试样例、来源、生成方式和期望结果。
- [testcases/qwen3_asr_testcases.json](./testcases/qwen3_asr_testcases.json)：结构化测试样例记录。

## 一句话总结

核心转换路径已经跑通；短音频端到端结果可对齐 PyTorch；剩余工作主要是把
C++ runtime 的长音频切块、拼接、KV cache、第二平台验证补齐。
