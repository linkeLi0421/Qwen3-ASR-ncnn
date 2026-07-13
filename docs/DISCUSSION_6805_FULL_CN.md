# Qwen3-ASR 转 ncnn：按分层验证更新的实验记录

实现分支：

```text
https://github.com/linkeLi0421/ncnn_llm/tree/qwen3-asr-export
```

记录仓库：

```text
https://github.com/linkeLi0421/Qwen3-ASR-ncnn
```

当前实现分支提交：

```text
68fd9af Align Qwen3-ASR forced language prompt
```

## 1. 当前结论

Qwen3-ASR 转 ncnn 的核心技术路径是可行的，但我现在不认为这个任务已经达到
“完成”状态。

原因是 Qwen3-ASR 不是一个可以整体直接转成 ncnn 静态图的简单模型。官方
`Qwen3ASRModel.transcribe()` 里包含音频读取、重采样、log-mel、切块、prompt
构造、audio token、generation wrapper、后处理等 Python pipeline。当前实现是把
神经网络部分拆成 ncnn 模块，再由 C++ runtime 补齐外围流程。

现在更合理的判断标准不是“某个 demo 音频能跑”，而是分层验证：

- 固定音频前处理；
- 三类中文 fixture 的端到端文本对齐；
- raw / clean / strict normalized / semantic normalized 分开记录；
- 模块级 summary 用来定位误差；
- 多平台 smoke 单独记录，不和 PyTorch accuracy parity 混为一个结论；
- 给出最小复测命令。

## 2. 当前拆分方式

当前导出并转换的 ncnn 模块：

- `audio_encoder`
- `text_embed`
- `text_backbone`
- `lm_head`

C++ runtime 负责：

- 读取 16 kHz mono PCM16 WAV；
- 生成 log-mel 特征；
- 构造 Qwen3-ASR prompt；
- 替换 audio token 对应的 audio embedding；
- greedy decode；
- 输出 text、JSON、mel summary 和部分模块 summary。

这说明转换路径可行，但最终文本一致性还需要继续按下面的分层表补证据。

## 3. 分层验证结果

新的测试入口在：

```text
tests/qwen3_asr
```

当前结果：

| 层级 | 当前状态 |
| --- | --- |
| 音频前处理 parity | ncnn C++ mel summary 和 PyTorch mel summary 已有；数值 diff 待补 |
| 三类中文 fixture | 本机 TTS fixture 已有：短中文、较长中文、中英数字混合；真人中文待补 |
| normalized text 对齐 | PyTorch baseline 已补；ncnn strict：1/3 通过；ncnn semantic：2/3 通过 |
| 模块级定位 | ncnn 侧 audio embedding、merged embedding、hidden、logits summary 已有；PyTorch fixed-chunk 模块 summary 已补；数值误差表待补 |
| 多平台 smoke | macOS CPU-only 已有；Linux CPU smoke 已有；Windows 待补 |
| 最小复测命令 | `qwen3_asr_main --model ... --audio-wav ... --text-out ... --json-out ... --dump-mel-summary ...` 已支持 |

三类中文 fixture，Linux VM CPU ncnn runtime 对比同机 PyTorch baseline：

| fixture | ncnn strict | ncnn semantic | PyTorch strict | PyTorch semantic | chunks | Linux RTF | Linux peak RSS | 说明 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `zh_short_tts` | PASS | PASS | PASS | PASS | 2 | 4.54 | 4738.9 MiB | forced language prompt 对齐后短中文输出和期望一致 |
| `zh_long_tts` | FAIL | FAIL | PASS | PASS | 9 | 3.60 | 4738.8 MiB | PyTorch 正确，ncnn 长音频仍有 `剪检查`、`对其` |
| `zh_mixed_tts` | FAIL | PASS | FAIL | FAIL | 4 | 3.72 | 4739.0 MiB | PyTorch 自身输出 `Open API`，fixture expected `OpenAI API` 过严；ncnn 英文缩写仍需后处理/解码契约确认 |

这里 strict normalized 是正式输出契约。semantic normalized 只用来标记缩写、空格、
标点等可解释差异，不能替代 strict 通过。

Linux VM 说明：RTX 4090 可以由 `nvidia-smi` 看到，但当前 Vulkan 只枚举到
`llvmpipe`，所以这里不是 4090/Vulkan 性能。原始 ncnn build 在这个 VM 上会在
`get_data_cache_size()` 触发 CPU cache parsing 的 `SIGFPE`；本次 Linux smoke 使用
本地临时 patched ncnn build 继续验证 Qwen3-ASR runtime。

这轮还修正了一个关键 pipeline 差异：C++ runtime 原来没有把 forced language prompt
里的 `language Chinese<asr_text>` 追加到 assistant prompt 后面，导致首个 chunk 的
`prompt_len` 是 48，而 PyTorch 是 51。修正后首个固定 chunk 的 `prompt_len` 对齐为
51，短中文首个 greedy token 与 PyTorch 同为 `100644`。这个修正降低了 Linux CPU
RTF，但没有解决长中文里的局部文字错误。

## 4. 已经支持的复测入口

示例命令：

```bash
qwen3_asr_main \
  --model <qwen3-asr-ncnn-runtime-dir> \
  --audio-wav sample.wav \
  --generate-from-features \
  --text-out out.txt \
  --json-out out.json \
  --dump-mel-summary mel.json
```

实现仓库里也有 fixture runner：

```bash
tests/qwen3_asr/run_fixture.sh \
  --binary <qwen3_asr_main> \
  --model <qwen3-asr-ncnn-runtime-dir> \
  --audio sample.wav \
  --out-dir out \
  --id sample \
  --threads 6 \
  --max-new-tokens 64 \
  --measure-memory
```

评估脚本会输出 strict normalized 和 semantic normalized 的对比报告。

## 5. 当前限制

- 还没有真实中文录音 fixture。
- 长中文 strict 文本仍未对齐。
- 当前 macOS 和 Linux smoke 都是 CPU runtime 结果，不代表 GPU/Vulkan 性能。
- Windows 最新分层 smoke 待补。
- 4090/Vulkan 需要单独验证设备可见性和输出可靠性。
- PyTorch 与 ncnn 的模块级数值误差表还没补齐；当前已有双方 summary。

## 6. 建议下一步

下一步不应继续扩大零散 demo，而应该补齐模块级定位和真实音频覆盖：

1. 对比 ncnn 和 PyTorch 的 `max_abs`、`mean_abs`、`p99_abs`、cosine similarity、
   top-k logits agreement。
2. 用真实中文录音补充短中文、长中文、中英数字混合三类 fixture。
3. 做 Windows smoke。
4. 单独排查 4090/Vulkan 设备可见性。
5. 将 Linux VM 上发现的 ncnn CPU cache parsing 问题整理为独立上游反馈或构建说明。

所以当前更准确的状态是：ncnn 转换和 C++ runtime 路径已经打通，但还需要 PyTorch
模块级对齐、真实音频验证和 GPU/Vulkan 验证，才能说这个 issue 真正完成。
