# Qwen3-ASR ncnn 验证记录

这个仓库只记录 `Qwen/Qwen3-ASR-0.6B` 转 ncnn 的验证结论、复测方法和当前缺口。

代码实现不在本仓库，见：

```text
https://github.com/linkeLi0421/ncnn_llm/tree/qwen3-asr-export
```

当前实现分支提交：

```text
828cd79 Support long Qwen3-ASR audio windows
```

## 当前结论

核心转换路径已经证明可行：Qwen3-ASR 可以拆成 `audio_encoder`、`text_embed`、
`text_backbone`、`lm_head` 等 ncnn 静态图模块，并由 C++ runtime 补齐 WAV 读取、
log-mel、prompt/audio token 组装、greedy decode 和输出解析。

最新一轮 Linux VM 验证修正了 forced language prompt：C++ runtime 现在按官方
PyTorch prompt 追加 `language Chinese<asr_text>`，首个固定 chunk 的 `prompt_len`
与 PyTorch 对齐为 51，短中文首 token 也对齐为 `100644`。

新的长音频实验确认了旧失败的根因：256-frame 独立 chunk 路径本身不等价于官方
`transcribe()`。官方对 18.78 秒样例会整段输入，产生 244 个 audio tokens，
`prompt_len=262`，旧 `text_seq_len=128` 根本装不下。重新导出
`audio_frames=1878`、`text_seq_len=512`，并启用 KV cache 后，长中文 fixture 已通过。

但这个 issue 还不能按“完成”处理：新路径只验证了长中文 TTS fixture，仍需补齐三类
fixture 的同一 long-window 复测、真实中文录音和非 CPU/Vulkan 平台结果。

## 当前分层结果

| 层级 | 当前状态 |
| --- | --- |
| 音频前处理 parity | ncnn C++ mel summary 和 PyTorch mel summary 已有；数值 diff 待补 |
| 三类中文 fixture | 本机 TTS fixture 已有：短中文、较长中文、中英数字混合；真人中文待补 |
| normalized text 对齐 | text128 旧基线：ncnn strict 1/3，semantic 2/3；long-window KV：`zh_long_tts` strict/semantic 通过 |
| 模块级定位 | ncnn 侧 audio embedding、merged embedding、hidden、logits summary 已有；PyTorch 对应模块 summary 已补；数值误差表待补 |
| 多平台 smoke | macOS CPU-only 已有；Linux CPU smoke 已有；Windows 待补 |
| 最小复测命令 | `qwen3_asr_main` 和 `tests/qwen3_asr/run_fixture.sh` 已支持 |

三类中文 fixture 当前结果，Linux VM CPU ncnn runtime 对比同机 PyTorch baseline：

| fixture | ncnn strict | ncnn semantic | PyTorch strict | PyTorch semantic | chunks | Linux RTF | Linux peak RSS | 结论 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `zh_short_tts` | PASS | PASS | PASS | PASS | 2 | 4.54 | 4738.9 MiB | prompt 对齐后短中文通过 |
| `zh_long_tts` | FAIL | FAIL | PASS | PASS | 9 | 3.60 | 4738.8 MiB | PyTorch 正确，ncnn 长音频仍有 `剪检查`、`对其` |
| `zh_mixed_tts` | FAIL | PASS | FAIL | FAIL | 4 | 3.72 | 4739.0 MiB | PyTorch 自身输出 `Open API`，fixture 期望 `OpenAI API` 过严；ncnn 英文缩写仍需后处理/解码契约确认 |

长音频修复实验：

| model/export | fixture | ncnn strict | chunks | prompt/audio tokens | wall time | peak RSS | 结论 |
| --- | --- | --- | ---: | --- | ---: | ---: | --- |
| `audio_frames=1878`, `text_seq_len=512`, KV cache | `zh_long_tts` | PASS | 1 | `262 / 244` | 70.29s | 19.31 GiB | 消除 `剪检查`、`对其`；输出与 PyTorch normalized text 对齐 |

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

1. 用 long-window KV 模型重跑短中文、长中文、中英数字混合三类 fixture。
2. 把 long-window export 命令和模型体积、内存代价整理到复测文档。
3. 用真实中文录音补充三类 fixture。
4. 做 Windows build smoke。
5. 单独验证 4090/Vulkan，不把 GPU 性能和 accuracy parity 混在一个结论里。
