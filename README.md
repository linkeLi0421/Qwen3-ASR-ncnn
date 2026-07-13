# Qwen3-ASR ncnn 转换记录

这个仓库用于记录 `Qwen/Qwen3-ASR-0.6B` 转 ncnn 的实验过程、当前结果、
失败原因和后续计划。

实际代码实现不在本仓库，代码分支在：

https://github.com/linkeLi0421/ncnn_llm/tree/qwen3-asr-export

当前实现分支最新提交：

```text
a6b650d Track Qwen3-ASR local memory smoke metrics
```

## 当前结论

目前已经验证：`Qwen/Qwen3-ASR-0.6B` 的核心神经网络模块可以通过
TorchScript + pnnx 转成 ncnn，并且 C++ ncnn runtime 可以在多个短音频样本上
和官方 PyTorch `Qwen3ASRModel.transcribe()` 输出一致。

但按照后续讨论中更严格的完成标准，这个 issue 还不能简单说“完成”。现在已经把
测评分成前处理、端到端文本、模块级定位、多平台 smoke 和最小复测命令几层，
并在实现分支加入了本机可重复的测试入口。

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

## 分层测评框架

新的本机测评入口在实现仓库：

https://github.com/linkeLi0421/ncnn_llm/tree/qwen3-asr-export/tests/qwen3_asr

当前本机结果：

| 层级 | 当前状态 |
| --- | --- |
| 音频前处理 parity | C++ mel summary 已输出；PyTorch summary 待补 |
| 三类中文 fixture | macOS TTS fixture 已有：短中文、较长中文、中文夹英文/数字/标点；真人中文待补 |
| normalized text 对齐 | strict：短中文通过，长中文和中英混合未通过；semantic：短中文和中英混合通过，长中文未通过 |
| 模块级误差 | 旧模块对比已有；新 ASR 分段 summary 已能输出 ncnn 侧 `audio_embedding`、`text_embeds`、`merged_embeds`、`hidden`、`logits`，PyTorch 对应 summary 待补 |
| 多平台 smoke | macOS arm64 CPU-only 已有；Linux/Windows 仍待补最新分层表；Vulkan/ARM 后续单列性能表 |
| 最小复测命令 | `qwen3_asr_main --model ... --audio-wav ... --text-out ... --json-out ...` 已支持；脚本已整理到 `tests/qwen3_asr/run_fixture.sh` |

三类中文 fixture 的当前结果：

| fixture | 类型 | strict normalized | semantic normalized | 说明 |
| --- | --- | --- | --- | --- |
| `zh_short_tts` | 短中文普通话 | PASS | PASS | 当前输出和期望一致 |
| `zh_long_tts` | 较长中文音频 | FAIL | FAIL | 输出到尾部，但仍有个别字词错误，例如 `检查/剪检查`、`对齐/对其` |
| `zh_mixed_tts` | 中文夹英文/数字/标点 | FAIL | PASS | strict 主要卡在 `OpenAI API` 的音近写法；semantic 规则下可认为语义通过 |

这个表比单个 demo 音频更接近上游提出的“ncnn 输出最终文本须与 PyTorch 原版一致”
要求。当前缺口集中在 PyTorch 侧同格式 summary、真人中文 fixture、严格文本对齐、
Linux/Windows 最新分层 smoke。

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
decode。下一阶段不应再只追 demo 跑通，而是补齐 PyTorch summary、真实中文样例、
严格 normalized text 对齐和 Linux/Windows 分层 smoke。
