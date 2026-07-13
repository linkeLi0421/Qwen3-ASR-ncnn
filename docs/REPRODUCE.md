# 复测命令

以下命令在实现仓库运行：

```bash
cd /Users/link/llk/ncnn_llm
```

## 生成本机中文 fixture

```bash
tests/qwen3_asr/make_chinese_fixtures.sh \
  --out-dir /Users/link/llk/test_audio/chinese_fixtures
```

## 运行单条 fixture

```bash
tests/qwen3_asr/run_fixture.sh \
  --binary /Users/link/llk/build/ncnn_llm-macos-qwen3-asr/qwen3_asr_main \
  --model /Users/link/llk/models/qwen3_asr_0_6b_runtime_text128 \
  --audio /Users/link/llk/test_audio/chinese_fixtures/zh_short_16k.wav \
  --out-dir /Users/link/llk/test_audio/chinese_fixtures \
  --id zh_short_current \
  --threads 6 \
  --max-new-tokens 64 \
  --measure-memory
```

输出：

```text
zh_short_current.txt
zh_short_current.json
zh_short_current_mel.json
zh_short_current_metrics.json
zh_short_current_time.txt
```

## 评估三类 fixture

```bash
python3 tests/qwen3_asr/evaluate_fixtures.py \
  --fixtures tests/qwen3_asr/fixtures.local.json \
  --json-out /Users/link/llk/test_audio/chinese_fixtures/eval_report.json \
  --markdown-out /Users/link/llk/test_audio/chinese_fixtures/eval_report.md
```

## 生成平台 smoke

```bash
python3 tests/qwen3_asr/collect_platform_smoke.py \
  --binary /Users/link/llk/build/ncnn_llm-macos-qwen3-asr/qwen3_asr_main \
  --model /Users/link/llk/models/qwen3_asr_0_6b_runtime_text128 \
  --eval-report /Users/link/llk/test_audio/chinese_fixtures/eval_report.json \
  --fixtures tests/qwen3_asr/fixtures.local.json \
  --threads 6 \
  --json-out /Users/link/llk/test_audio/chinese_fixtures/platform_smoke.json \
  --markdown-out /Users/link/llk/test_audio/chinese_fixtures/platform_smoke.md
```

## 最小 CLI 形态

最终希望上游用户能用类似命令复测：

```bash
qwen3_asr_main \
  --model <qwen3-asr-ncnn-runtime-dir> \
  --audio-wav sample.wav \
  --generate-from-features \
  --text-out out.txt \
  --json-out out.json \
  --dump-mel-summary mel.json
```
