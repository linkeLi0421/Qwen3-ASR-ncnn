# Qwen3-ASR ncnn 转换记录

这个仓库用于记录 `Qwen/Qwen3-ASR-0.6B` 转 ncnn 的实验过程、当前结果、
失败原因和后续计划。

实际代码实现不在本仓库，代码分支在：

https://github.com/linkeLi0421/ncnn_llm/tree/qwen3-asr-export

当前实现分支最新提交：

```text
89394d2 Gate Qwen3-ASR timing logs behind flag
```

## 当前结论

目前已经验证：`Qwen/Qwen3-ASR-0.6B` 的核心神经网络模块可以通过
TorchScript + pnnx 转成 ncnn，并且 C++ ncnn runtime 可以在多个短音频样本上
和官方 PyTorch `Qwen3ASRModel.transcribe()` 输出一致。

主要实验结果：

| 样本 | runtime | 时长 | PyTorch | ncnn | 结论 |
| --- | --- | ---: | --- | --- | --- |
| `pdx_voice` | text128 + CMake | 4.95s | `This is a test of me recording my voice.` | `This is a test of me recording my voice.` | 通过 |
| `hello_world` | text64 | 1.36s | `Hello world.` | `Hello world.` | 通过 |
| `natural_zero` | text64 | 0.64s | `Zero.` | `Zero.` | 通过 |
| `digit_five` | text64 | 0.42s | `Five.` | `Five.` | 通过 |
| `long_text_numbers_fast` | text64 | 2.22s | `One two three four five six seven eight nine ten eleven twelve thirteen fourteen.` | `One two three four five six seven eight nine ten eleven twelve thirteen` | 截断 |
| `long_text_numbers_fast` | text128 | 2.22s | `One two three four five six seven eight nine ten eleven twelve thirteen fourteen.` | `One two three four five six seven eight nine ten eleven twelve thirteen fourteen.` | 通过 |
| `long_text_numbers_fast` | text48 + KV cache | 2.22s | text128 static | `One two three four five six seven eight nine ten eleven twelve thirteen fourteen.` | 通过 |
| `long_text_numbers_30` | text128 / text48 + KV cache | 7.00s | 合成 one 到 thirty | 两者输出一致，生成 77 tokens | 当前基线通过 |

仍未完全解决的是长音频处理。当前已加入 32-frame overlap chunking 和词级
suffix/prefix 去重拼接，但还不等价于官方 Python pipeline。

正式 CMake 构建已经加入实现分支，并在 Linux/RTX 4090 VM 上验证通过。
用正式 CMake build 出来的 `qwen3_asr_main` 跑 `pdx_voice_16k.wav`，最终输出：

```text
text=This is a test of me recording my voice.
```

第二平台也已验证：macOS 15.7.4 / Apple M1 Pro / arm64 / CPU-only。使用本机
源码构建的 ncnn CPU-only 和 `qwen3_asr_main`，同样可以跑通
`pdx_voice_16k.wav`：

```text
text=This is a test of me recording my voice.
```

长文本限制也已通过重新导出 `text_seq_len=128` 缓解。快速数字语音样例中，
text64 会截断在 `... twelve thirteen`，text128 可以完整输出到
`... thirteen fourteen.`。实验性 KV cache 路径使用 text48 prefill，也能输出
完整 long_text_numbers_fast；在 7 秒 one-to-thirty 压力样例上，KV48 与
text128 输出一致，生成 77 个 token。

KV cache timing 也已补充：`qwen3_asr_main` 已加入 `--timing` 选项；在
Linux VM 的 CPU ncnn runtime 上，KV48 相比
static text128 的 total measured speedup 约为 3.40x 到 6.22x。RTX 4090 GPU/Vulkan
加速仍未验证，因为当前 `--vulkan` 只枚举到 `llvmpipe` 软件 Vulkan device。

长音频 overlap/stitching 回归结果：

```text
pdx_voice:               This is a test of me recording my voice.
long_text_numbers_fast:  One two three four five six seven eight nine ten eleven twelve thirteen fourteen.
long_digit_five:         By by by by by by five, five, five.  # 人工重复压力样例，仍不作为通过样本
```

## 仓库文件

- [NOTES.md](./NOTES.md)：完整中文技术记录。
- [DISCUSSION_REPORT.md](./DISCUSSION_REPORT.md)：给 `Tencent/ncnn` discussion 6805 的中文更新。
- [DISCUSSION_6805_FULL_CN.md](./DISCUSSION_6805_FULL_CN.md)：可直接复制覆盖原 discussion 文章的完整中文稿。
- [UPSTREAM_SUMMARY.md](./UPSTREAM_SUMMARY.md)：给上游 issue/discussion 使用的英文摘要。
- [FAILURE_RETRY_LOG.md](./FAILURE_RETRY_LOG.md)：失败、误判、重试和修复记录。
- [TESTCASES.md](./TESTCASES.md)：测试样例、来源、生成方式和期望结果。
- [testcases/qwen3_asr_testcases.json](./testcases/qwen3_asr_testcases.json)：结构化测试样例记录。

## 一句话总结

核心转换路径已经跑通；短音频端到端结果可对齐 PyTorch；C++ runtime 已加入
实验性 KV cache 路径，并在短样例批量测试和当前长输出压力样例上对齐默认静态
decode。剩余工作主要是排查 KV64/KV128 导出矩阵、继续改进长音频切块/拼接，
并继续扩大平台和样例覆盖。
