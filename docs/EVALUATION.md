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

## 当前结果

| fixture | strict | semantic | chunks | RTF | peak RSS | notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `zh_short_tts` | PASS | PASS | 2 | 10.71 | 3485.5 MiB | 输出和期望一致 |
| `zh_long_tts` | FAIL | FAIL | 9 | 11.36 | 3463.8 MiB | 输出到尾部，但有 `剪检查`、`对其` |
| `zh_mixed_tts` | FAIL | PASS | 4 | 10.33 | 3448.0 MiB | `OpenAI API` strict 不一致，semantic 可归一 |

strict normalized 是正式输出契约。semantic normalized 只用于记录缩写、空格、标点
等可解释差异，不能替代 strict 通过。

## 当前报告文件

本机生成的报告在：

```text
/Users/link/llk/test_audio/chinese_fixtures/eval_report.json
/Users/link/llk/test_audio/chinese_fixtures/eval_report.md
/Users/link/llk/test_audio/chinese_fixtures/platform_smoke.json
/Users/link/llk/test_audio/chinese_fixtures/platform_smoke.md
```

这些本地输出不提交到 notes 仓库；公开复测入口在实现仓库的 `tests/qwen3_asr`。

## 旧测试的处理

旧的英文 demo、KV timing、text64/text128 压力样例已经移动到 `archive/`。
它们仍有历史价值，但不再定义当前任务是否完成。
