# Qwen3-ASR ncnn 验证记录

这个仓库只记录 `Qwen/Qwen3-ASR-0.6B` 转 ncnn 的验证结论、复测方法和当前缺口。

代码实现不在本仓库，见：

```text
https://github.com/linkeLi0421/ncnn_llm/tree/qwen3-asr-export
```

当前实现分支提交：

```text
3109bf9 Document Qwen3-ASR local smoke memory metrics
```

## 当前结论

核心转换路径已经证明可行：Qwen3-ASR 可以拆成 `audio_encoder`、`text_embed`、
`text_backbone`、`lm_head` 等 ncnn 静态图模块，并由 C++ runtime 补齐 WAV 读取、
log-mel、prompt/audio token 组装、greedy decode 和输出解析。

但这个 issue 还不能按“完成”处理。新的完成标准是分层验证，而不是只看一个 demo
音频能否跑通。

## 当前分层结果

| 层级 | 当前状态 |
| --- | --- |
| 音频前处理 parity | C++ mel summary 已有；PyTorch 同 schema summary 待补 |
| 三类中文 fixture | 本机 TTS fixture 已有：短中文、较长中文、中英数字混合；真人中文待补 |
| normalized text 对齐 | strict：1/3 通过；semantic：2/3 通过 |
| 模块级定位 | ncnn 侧 audio embedding、merged embedding、hidden、logits summary 已有；PyTorch 对应 summary 待补 |
| 多平台 smoke | macOS CPU-only 已有；Linux/Windows 最新分层 smoke 待补 |
| 最小复测命令 | `qwen3_asr_main` 和 `tests/qwen3_asr/run_fixture.sh` 已支持 |

三类中文 fixture 当前结果：

| fixture | strict | semantic | chunks | RTF | peak RSS | 结论 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `zh_short_tts` | PASS | PASS | 2 | 10.71 | 3485.5 MiB | 短中文通过 |
| `zh_long_tts` | FAIL | FAIL | 9 | 11.36 | 3463.8 MiB | 长中文仍有词错误，例如 `剪检查`、`对其` |
| `zh_mixed_tts` | FAIL | PASS | 4 | 10.33 | 3448.0 MiB | `OpenAI API` 缩写写法 strict 不一致 |

## 文档结构

- [docs/CURRENT_STATUS.md](./docs/CURRENT_STATUS.md)：当前完成度和缺口。
- [docs/EVALUATION.md](./docs/EVALUATION.md)：新的分层测评方法和结果。
- [docs/REPRODUCE.md](./docs/REPRODUCE.md)：最小复测命令。
- [docs/DISCUSSION_6805_FULL_CN.md](./docs/DISCUSSION_6805_FULL_CN.md)：可复制到上游 discussion 的中文稿。
- [archive/](./archive/)：旧实验记录、旧测试表和失败重试日志。只作为历史背景，不再作为当前完成标准。

## 下一步

本机已经能做的主要工作已经完成。继续推进需要 Linux/PyTorch 环境：

1. 跑原版 PyTorch Qwen3-ASR，生成同 schema 的 text、mel、module summary。
2. 对比 ncnn 与 PyTorch 的 encoder/projector/decoder 首轮 logits 误差。
3. 用真实中文录音补充三类 fixture。
4. 做 Linux/Windows build smoke。
5. 单独验证 4090/Vulkan，不把 GPU 性能和 accuracy parity 混在一个结论里。
