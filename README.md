# Qwen3-ASR ncnn 验证记录

这个仓库只记录 `Qwen/Qwen3-ASR-0.6B` 转 ncnn 的验证结论、复测方法和当前缺口。

代码实现不在本仓库，见：

```text
https://github.com/linkeLi0421/ncnn_llm/tree/qwen3-asr-export
```

当前实现分支提交：

```text
5cf4fb6 Support Linux memory metrics for Qwen3-ASR fixtures
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
| 音频前处理 parity | ncnn C++ mel summary 和 PyTorch mel summary 已有；数值 diff 待补 |
| 三类中文 fixture | 本机 TTS fixture 已有：短中文、较长中文、中英数字混合；真人中文待补 |
| normalized text 对齐 | PyTorch baseline 已补；ncnn strict：1/3 通过；ncnn semantic：2/3 通过 |
| 模块级定位 | ncnn 侧 audio embedding、merged embedding、hidden、logits summary 已有；PyTorch 对应模块误差表待补 |
| 多平台 smoke | macOS CPU-only 已有；Linux CPU smoke 已有；Windows 待补 |
| 最小复测命令 | `qwen3_asr_main` 和 `tests/qwen3_asr/run_fixture.sh` 已支持 |

三类中文 fixture 当前结果，Linux VM CPU ncnn runtime 对比同机 PyTorch baseline：

| fixture | ncnn strict | ncnn semantic | PyTorch strict | PyTorch semantic | chunks | Linux RTF | Linux peak RSS | 结论 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `zh_short_tts` | PASS | PASS | PASS | PASS | 2 | 5.79 | 4739.0 MiB | 短中文通过 |
| `zh_long_tts` | FAIL | FAIL | PASS | PASS | 9 | 4.72 | 4738.5 MiB | PyTorch 正确，ncnn 长音频仍有 `剪检查`、`对其` |
| `zh_mixed_tts` | FAIL | PASS | FAIL | FAIL | 4 | 4.54 | 4739.1 MiB | PyTorch 自身输出 `Open API`，fixture 期望 `OpenAI API` 过严；ncnn 还会把英文缩写拆散 |

Linux VM 说明：RTX 4090 可由 `nvidia-smi` 看到，但当前 Vulkan 只枚举到
`llvmpipe`，所以上表不是 4090/Vulkan 性能。原始 ncnn build 在该 VM 上触发
`get_data_cache_size()` 的 CPU cache parsing `SIGFPE`；Linux smoke 使用的是本地
临时 patched ncnn build，用于继续验证 Qwen3-ASR runtime。

## 文档结构

- [docs/CURRENT_STATUS.md](./docs/CURRENT_STATUS.md)：当前完成度和缺口。
- [docs/EVALUATION.md](./docs/EVALUATION.md)：新的分层测评方法和结果。
- [docs/REPRODUCE.md](./docs/REPRODUCE.md)：最小复测命令。
- [docs/DISCUSSION_6805_FULL_CN.md](./docs/DISCUSSION_6805_FULL_CN.md)：可复制到上游 discussion 的中文稿。
- [archive/](./archive/)：旧实验记录、旧测试表和失败重试日志。只作为历史背景，不再作为当前完成标准。

## 下一步

当前还能继续推进的主要工作：

1. 对比 ncnn 与 PyTorch 的 encoder/projector/decoder 首轮 logits 误差。
2. 将 VM 上发现的 ncnn CPU cache parsing 问题整理为独立上游反馈或本地构建说明。
3. 用真实中文录音补充三类 fixture。
4. 做 Windows build smoke。
5. 单独验证 4090/Vulkan，不把 GPU 性能和 accuracy parity 混在一个结论里。
