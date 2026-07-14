# Qwen3-ASR ncnn 验证记录

这个仓库只记录 `Qwen/Qwen3-ASR-0.6B` 转 ncnn 的验证结论、复测方法和当前缺口。

代码实现不在本仓库，见：

```text
https://github.com/linkeLi0421/ncnn_llm/tree/qwen3-asr-export
```

当前实现分支提交：

```text
35174d2 Align Qwen3-ASR module summaries
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

但这个 issue 还不能按“完成”处理：long-window KV 已完成三类 TTS fixture 复测，
仍需补真实中文录音和非 CPU/Vulkan 平台结果。

## 当前分层结果

| 层级 | 当前状态 |
| --- | --- |
| 音频前处理 parity | ncnn C++ mel summary 和 PyTorch mel summary 已有；数值 diff 待补 |
| 三类中文 fixture | 本机 TTS fixture 已有：短中文、较长中文、中英数字混合；真人中文待补 |
| normalized text 对齐 | text128 旧基线：ncnn strict 1/3，semantic 2/3；long-window KV：ncnn strict 2/3，semantic 2/3；混合样例 PyTorch 也 strict fail |
| 模块级定位 | long-window 静态路径的 ncnn/PyTorch summary 已同 shape 对齐，首 token 一致；raw tensor 数值误差表待补 |
| 多平台 smoke | macOS CPU-only 已有；Linux CPU smoke 已有；Windows 待补 |
| 最小复测命令 | `qwen3_asr_main` 和 `tests/qwen3_asr/run_fixture.sh` 已支持 |

三类中文 fixture 当前结果，Linux VM CPU ncnn runtime 对比同机 PyTorch baseline：

| fixture | ncnn strict | ncnn semantic | PyTorch strict | PyTorch semantic | chunks | Linux RTF | Linux peak RSS | 结论 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `zh_short_tts` | PASS | PASS | PASS | PASS | 2 | 4.54 | 4738.9 MiB | prompt 对齐后短中文通过 |
| `zh_long_tts` | FAIL | FAIL | PASS | PASS | 9 | 3.60 | 4738.8 MiB | PyTorch 正确，ncnn 长音频仍有 `剪检查`、`对其` |
| `zh_mixed_tts` | FAIL | PASS | FAIL | FAIL | 4 | 3.72 | 4739.0 MiB | PyTorch 自身输出 `Open API`，fixture 期望 `OpenAI API` 过严；ncnn 英文缩写仍需后处理/解码契约确认 |

long-window KV 复测：

| fixture | ncnn strict | ncnn semantic | PyTorch strict | PyTorch semantic | chunks | RTF | wall time | peak RSS | 结论 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `zh_short_tts` | PASS | PASS | PASS | PASS | 1 | 10.00 | 40.61s | 19.37 GiB | 短中文仍通过 |
| `zh_long_tts` | PASS | PASS | PASS | PASS | 1 | 3.82 | 72.98s | 19.37 GiB | 消除 `剪检查`、`对其` |
| `zh_mixed_tts` | FAIL | FAIL | FAIL | FAIL | 1 | 5.66 | 52.39s | 19.38 GiB | ncnn 和 PyTorch 均输出 `OpenAPI`，不是 ncnn 独有失败 |

补充复测：修正 debug summary 只统计真实 `prompt_len=262` 后，ncnn 和 PyTorch
static module summary 已对齐。三条 fixture 都是 `input_features=[1,128,1878]`、
`audio_embedding=[244,1024]`、`prompt_len=262`，首个 greedy token 分别为
`100644`、`43288`、`100644`，与 ncnn 一致。注意这里的 PyTorch module summary
刻意使用全 1 feature mask 来模拟 ncnn 静态图；官方端到端 PyTorch baseline 仍使用
原版 `transcribe()`。

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

1. 补 raw tensor 级模块误差：`max_abs`、`mean_abs`、`p99_abs`、cosine、top-k logits agreement。
2. 把 long-window export 命令和模型体积、内存代价整理到复测文档。
3. 用真实中文录音补充三类 fixture。
4. 做 Windows build smoke。
5. 单独验证 4090/Vulkan，不把 GPU 性能和 accuracy parity 混在一个结论里。
