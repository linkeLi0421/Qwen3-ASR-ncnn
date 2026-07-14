# 分层测评

实现仓库：

```text
https://github.com/linkeLi0421/ncnn_llm/tree/qwen3-asr-export/tests/qwen3_asr
```

## 测评目标

把“ncnn 输出最终文本须与 PyTorch 原版一致”拆成几层，避免只测到 demo 音频：

1. 固定音频前处理。
2. 三类中文 fixture 的端到端文本对齐。
3. raw、clean、strict normalized、semantic normalized 分开记录。
4. 模块级 summary 用来定位误差。
5. 多平台 smoke 只记录可运行性和资源，不和 PyTorch accuracy parity 混为一个结论。

## Fixture

| fixture | 类型 | expected |
| --- | --- | --- |
| `zh_short_tts` | 短中文普通话 | `今天天气很好，我们一起测试语音识别。` |
| `zh_long_tts` | 较长中文音频 | 包含前处理、模块误差、文本对齐、smoke test 的长句 |
| `zh_mixed_tts` | 中文夹英文/数字/标点 | 包含 `OpenAI API`、日期和订单编号 |

音频由 macOS `say` 生成，再转为 16 kHz mono PCM16 WAV。它们是本机稳定 fixture，
不是最终真人语音覆盖。

## 当前主结果

Linux VM 上使用同一批 fixture 跑原版 PyTorch baseline 和 ncnn CPU runtime：

| fixture | ncnn strict | ncnn semantic | PyTorch strict | PyTorch semantic | chunks | Linux RTF | Linux peak RSS | 说明 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `zh_short_tts` | PASS | PASS | PASS | PASS | 2 | 4.54 | 4738.9 MiB | forced language prompt 对齐后短中文通过 |
| `zh_long_tts` | FAIL | FAIL | PASS | PASS | 9 | 3.60 | 4738.8 MiB | PyTorch 正确，ncnn 长音频仍有 `剪检查`、`对其` |
| `zh_mixed_tts` | FAIL | PASS | FAIL | FAIL | 4 | 3.72 | 4739.0 MiB | PyTorch 自身输出 `Open API`，fixture expected `OpenAI API` 过严；ncnn 英文缩写仍需后处理/解码契约确认 |

这个结果把问题拆开了：

- `zh_short_tts` 证明短中文端到端路径可用。
- `zh_long_tts` 是当前最明确的 ncnn runtime/chunk stitching 缺口，因为 PyTorch baseline
  对同一音频通过。
- `zh_mixed_tts` 不能作为纯 ncnn 转换失败结论，因为 PyTorch baseline 本身也不满足
  这个 fixture 的 strict expected。

本轮还修正了 C++ prompt 构造：当指定 `--language Chinese` 时，runtime 会按官方
PyTorch prompt 追加 `language Chinese<asr_text>`。首个固定 chunk 的 `prompt_len`
从旧实现的 48 对齐到 PyTorch 的 51，短中文首个 greedy token 与 PyTorch 同为
`100644`。

PyTorch 模块 summary 也已经生成，覆盖 fixed overlap chunk 上的 audio embedding、
merged embedding、hidden、logits 和 selected logits。当前仍缺的是把这些 summary
与 ncnn summary 做逐项数值误差表。

## Long-Window KV 结果

进一步实验显示，`zh_long_tts` 失败并不是单纯的 ncnn 算子误差。把同样的 9 个
256-frame chunk 交给 PyTorch 原版 `transcribe()` 逐块识别，也会得到 `主要用于剪`
和 `文本对其`。官方 full-audio `transcribe()` 对这个 18.78 秒样例不会切成 2.56 秒
短窗，而是整段输入：

```text
input_features: [1, 128, 1878]
audio_features: [244, 1024]
prompt_len: 262
```

因此旧 `text_seq_len=128` 模型无法表示官方路径。重新导出：

```text
audio_frames=1878
text_seq_len=512
kv_cache_len=511
```

并修正 KV prefill cache 裁剪后，Linux CPU ncnn 结果为：

| fixture | mode | strict | semantic | chunks | wall time | peak RSS | text |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `zh_long_tts` | full1878 text512 KV | PASS | PASS | 1 | 70.29s | 19.31 GiB | `这是一段较长的中文语音测试。我们希望模型能够稳定地识别连续的句子，并且在音频变长以后仍然保持合理的文本输出。这个样例主要用于检查切块拼接以及端到端文本对齐是否可靠。` |

KV 修复点：pnnx trace 的 prefill KV cache 是静态 512 长度，但真实 prompt_len 是 262。
runtime 必须在 prefill 后把 K/V cache 裁到真实 prompt 长度，否则 decode 会 attend 到
padding cache，第二个 token 开始就会错。

## macOS 本机结果

| fixture | strict | semantic | chunks | RTF | peak RSS | notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `zh_short_tts` | PASS | PASS | 2 | 10.71 | 3485.5 MiB | 输出和期望一致 |
| `zh_long_tts` | FAIL | FAIL | 9 | 11.36 | 3463.8 MiB | 输出到尾部，但有 `剪检查`、`对其` |
| `zh_mixed_tts` | FAIL | PASS | 4 | 10.33 | 3448.0 MiB | `OpenAI API` strict 不一致，semantic 可归一 |

strict normalized 是正式输出契约。semantic normalized 只用于记录缩写、空格、标点
等可解释差异，不能替代 strict 通过。

## Linux VM 环境记录

有效 smoke：

```text
OS: Linux VM
GPU: RTX 4090 24GB present in nvidia-smi
ncnn runtime: CPU path, threads=8
model: qwen3_asr_0_6b_runtime_text128
PyTorch model: Qwen/Qwen3-ASR-0.6B
```

限制：

- 当前 Vulkan 只枚举到 `llvmpipe`，不是 RTX 4090 Vulkan timing。
- 原始 ncnn build 在该 VM 上触发 `get_data_cache_size()` 的 `SIGFPE`。本次 Linux
  ncnn smoke 使用临时 patched ncnn build 绕过 CPU cache parsing 除零问题。

## 当前报告文件

本机和 VM 生成的报告在：

```text
/Users/link/llk/test_audio/chinese_fixtures/eval_report.json
/Users/link/llk/test_audio/chinese_fixtures/eval_report.md
/Users/link/llk/test_audio/chinese_fixtures/platform_smoke.json
/Users/link/llk/test_audio/chinese_fixtures/platform_smoke.md
/data/results/qwen3_asr/eval_vm_linux_patched.json
/data/results/qwen3_asr/eval_vm_linux_patched.md
/data/results/qwen3_asr/eval_vm_forced_prompt.json
/data/results/qwen3_asr/eval_vm_forced_prompt.md
/data/results/qwen3_asr/linux_ncnn_full1878_text512_kv_full/zh_long_tts.json
/data/results/qwen3_asr/linux_ncnn_full1878_text512_kv_full/zh_long_tts_time.txt
```

这些输出不提交到 notes 仓库；公开复测入口在实现仓库的 `tests/qwen3_asr`。

## 旧测试的处理

旧的英文 demo、KV timing、text64/text128 压力样例已经移动到 `archive/`。
它们仍有历史价值，但不再定义当前任务是否完成。
