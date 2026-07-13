# 历史归档

这个目录保存早期实验记录、旧测试样例和旧 discussion 草稿。

这些文件不再定义当前完成标准。当前有效文档在：

```text
../README.md
../docs/CURRENT_STATUS.md
../docs/EVALUATION.md
../docs/REPRODUCE.md
```

归档原因：

- 早期主要依赖英文 demo、text64/text128、KV timing、人工长音频压力样例。
- 这些结果对调试有价值，但不适合作为 “Qwen3-ASR ncnn 输出与 PyTorch 原版一致”
  的最终证据。
- 当前主线已经改为 `tests/qwen3_asr` 分层测评。

文件说明：

| 文件 | 说明 |
| --- | --- |
| `NOTES.md` | 早期完整技术流水账 |
| `TESTCASES.md` | 旧测试样例表 |
| `qwen3_asr_testcases.json` | 旧结构化样例记录 |
| `FAILURE_RETRY_LOG.md` | 失败和重试记录 |
| `DISCUSSION_REPORT.md` | 旧 discussion 更新 |
| `UPSTREAM_SUMMARY.md` | 旧英文摘要 |
