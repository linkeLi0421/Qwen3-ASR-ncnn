# 测试样例记录

这个文件记录当前用于比较官方 PyTorch Qwen3-ASR 和 ncnn C++ runtime 的测试样例。

本仓库默认不保存 WAV 二进制文件，只保存来源、生成方式和期望输出，避免 notes
仓库变大。结构化版本见：

```text
testcases/qwen3_asr_testcases.json
```

## 测试方法

PyTorch 使用：

```python
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "/data/qwen3-asr-ncnn/models/Qwen3-ASR-0.6B",
    device_map="cuda",
)

result = model.transcribe(wav, language="English")
print(result)
```

ncnn 使用：

```bash
/data/qwen3-asr-ncnn/build/ncnn_llm_cmake/qwen3_asr_main \
  --model /data/qwen3-asr-ncnn/models/qwen3_asr_0_6b_runtime_text128 \
  --audio-wav "$wav" \
  --generate-from-features \
  --language English \
  --chunk-overlap-frames 32 \
  --max-new-tokens 64
```

输入 WAV 要求：

- 16 kHz
- mono
- PCM16 WAV

通用转换命令：

```bash
ffmpeg -y -i input.wav -ar 16000 -ac 1 output_16k.wav
```

## 当前样例表

| 样例 | 类型 | runtime | 时长 | PyTorch | ncnn | 结论 |
| --- | --- | --- | ---: | --- | --- | --- |
| `pdx_voice` | 真实短语音 | text128 + CMake | 4.95s | `This is a test of me recording my voice.` | `This is a test of me recording my voice.` | 通过 |
| `pdx_voice_kv_cache` | 真实短语音 | text48 + experimental KV cache | 4.95s | `This is a test of me recording my voice.` | `This is a test of me recording my voice.` | 通过 |
| `kv_cache_batch_short` | KV cache 批量对比 | text128 static vs text48 KV | 0.42-4.95s | 默认静态 decode | 6/6 输出一致 | 通过 |
| `hello_world` | 合成短语音 | text64 | 1.36s | `Hello world.` | `Hello world.` | 通过 |
| `natural_zero` | 真实短 digit | text64 | 0.64s | `Zero.` | `Zero.` | 通过 |
| `digit_five` | 真实短 digit | text64 | 0.42s | `Five.` | `Five.` | 通过 |
| `long_text_numbers_fast` | 合成长文本压力样例 | text64 | 2.22s | `One two three four five six seven eight nine ten eleven twelve thirteen fourteen.` | `One two three four five six seven eight nine ten eleven twelve thirteen` | 截断 |
| `long_text_numbers_fast` | 合成长文本压力样例 | text128 | 2.22s | `One two three four five six seven eight nine ten eleven twelve thirteen fourteen.` | `One two three four five six seven eight nine ten eleven twelve thirteen fourteen.` | 通过 |
| `long_text_numbers_fast_kv` | KV 长输出 | text48 + KV cache | 2.22s | text128 static | `One two three four five six seven eight nine ten eleven twelve thirteen fourteen.` | 通过 |
| `long_text_numbers_30` | 合成长输出/拼接压力样例 | text64/text128/KV48 | 7.00s | 合成 one 到 thirty | text128 与 KV48 一致；text64 截断 | 部分通过 |
| `long_digit_five` | 人工重复压力样例 | text128 overlap chunking | 16.97s | `By by` | `By by by by by by five, five, five.` | 不作为语义正确性 benchmark |

## KV cache 批量对比

测试目的：验证实验性 `text_prefill_kv` + `text_decode_kv` 路径是否和默认静态 decode
保持一致。这里对比的是两个 ncnn runtime 路径，不代表每条样例都重新跑了 PyTorch。

| 样本 | 时长 | 默认静态 decode | KV cache decode | 结论 |
| --- | ---: | --- | --- | --- |
| `pdx_voice_16k.wav` | 4.95s | `This is a test of me recording my voice.` | `This is a test of me recording my voice.` | 对齐 |
| `pdx_voice-note_16k.wav` | 1.40s | `啊。` | `啊。` | 对齐 |
| `0_jackson_0_16k.wav` | 0.64s | `Zero.` | `Zero.` | 对齐 |
| `1_jackson_0_16k.wav` | 0.52s | `一。` | `一。` | 对齐 |
| `natural_digit_0_16k.wav` | 0.64s | `Zero.` | `Zero.` | 对齐 |
| `random_digit_5_jackson_0_16k.wav` | 0.42s | `Five.` | `Five.` | 对齐 |

原始 `0_jackson_0.wav` 和 `1_jackson_0.wav` 是 8 kHz；runtime 当前只接受 16 kHz WAV，
所以测试前转成了 `0_jackson_0_16k.wav` 和 `1_jackson_0_16k.wav`。

## 长输出和 text_seq_len 对比

`long_text_numbers_fast_16k.wav` 用来验证单 chunk 长文本容量：

| runtime | `text_seq_len` | 输出 | 结论 |
| --- | ---: | --- | --- |
| static | 64 | `One two three four five six seven eight nine ten eleven twelve thirteen` | 截断 |
| static | 128 | `One two three four five six seven eight nine ten eleven twelve thirteen fourteen.` | 通过 |
| KV cache | 48 | `One two three four five six seven eight nine ten eleven twelve thirteen fourteen.` | 通过 |

`long_text_numbers_30_16k.wav` 是 7.00 秒合成 one 到 thirty，跨 3 个 chunk：

| runtime | chunk 数 | 生成 token 数 | 输出摘要 | 结论 |
| --- | ---: | ---: | --- | --- |
| static text64 | 3 | 48 | `One two ... thirteen 16, 17, 18, 1 twenty-four ... twenty` | 截断/拼接不完整 |
| static text128 | 3 | 77 | `One two ... sixteen 16, 17, 18, 19, 20, 21, 22, 23 twenty-four ... thirty.` | 当前基线通过 |
| KV cache text48 | 3 | 77 | 与 static text128 完全一致 | 当前基线通过 |

KV64 导出尝试：

```bash
python3 export/qwen3_asr_export.py \
  --model-id Qwen/Qwen3-ASR-0.6B-hf \
  --out-dir /data/qwen3-asr-ncnn/models/qwen3_asr_0_6b_runtime_kv64 \
  --device cuda --dtype fp32 \
  --text-seq-len 64 --export-kv-cache --kv-cache-len 64 \
  --convert-ncnn --pnnx-bin /data/qwen3-asr-ncnn/build/pnnx/src/pnnx
```

当前 VM 上该导出长时间停在模型加载/初始化阶段，输出目录为空，未形成可测试的
KV64 模型包；这需要后续单独排查导出环境。

## 样例来源和生成方式

### `pdx_voice`

来源：

```text
https://github.com/pdx-cs-sound/wavs/blob/main/voice.wav
```

说明：真实短语音，许可证见上游仓库。这个样例是当前最有价值的端到端验证样例。

转换：

```bash
ffmpeg -y -i voice.wav -ar 16000 -ac 1 pdx_voice_16k.wav
```

期望结果：

```text
PyTorch: This is a test of me recording my voice.
ncnn:    This is a test of me recording my voice.
```

### `hello_world`

来源：本地用 `espeak-ng` 合成。

生成：

```bash
espeak-ng -w hello_world.wav "hello world"
ffmpeg -y -i hello_world.wav -ar 16000 -ac 1 hello_world_16k.wav
```

期望结果：

```text
PyTorch: Hello world.
ncnn:    Hello world.
```

### `natural_zero`

来源：

```text
https://github.com/Jakobovski/free-spoken-digit-dataset/blob/master/recordings/0_jackson_0.wav
```

转换：

```bash
ffmpeg -y -i 0_jackson_0.wav -ar 16000 -ac 1 natural_digit_0_16k.wav
```

期望结果：

```text
PyTorch: Zero.
ncnn:    Zero.
```

### `digit_five`

来源：

```text
https://github.com/Jakobovski/free-spoken-digit-dataset/blob/master/recordings/5_jackson_0.wav
```

转换：

```bash
ffmpeg -y -i 5_jackson_0.wav -ar 16000 -ac 1 random_digit_5_jackson_0_16k.wav
```

期望结果：

```text
PyTorch: Five.
ncnn:    Five.
```

### `long_digit_five`

来源：把 `digit_five` 重复 40 次得到的人工压力样例。

说明：这个样例不适合作为 ASR 语义正确性 benchmark。官方 PyTorch 也没有输出
重复的 `Five.`，而是输出 `By by`。它主要用于暴露当前 ncnn runtime 的长音频
chunk stitching 限制。

生成：

```bash
: > long_digit_5_x40_concat.txt
for i in $(seq 1 40); do
  printf "file '%s'\n" "random_digit_5_jackson_0_16k.wav" >> long_digit_5_x40_concat.txt
done

ffmpeg -y -f concat -safe 0 \
  -i long_digit_5_x40_concat.txt \
  -c copy long_digit_5_x40_16k.wav
```

当前结果：

```text
PyTorch: By by
ncnn:    By by by by by by five, five, five.
```

结论：这是长音频切块拼接问题，不是核心 ncnn 模块转换失败。

### `long_text_numbers_fast`

来源：本地用 `espeak-ng` 合成，用于验证 `text_seq_len` 是否限制生成文本长度。

生成：

```bash
espeak-ng -s 340 -w long_text_numbers_fast.wav \
  "one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
ffmpeg -y -i long_text_numbers_fast.wav \
  -ar 16000 -ac 1 long_text_numbers_fast_16k.wav
```

PyTorch 结果：

```text
One two three four five six seven eight nine ten eleven twelve thirteen fourteen.
```

text64 ncnn 结果：

```text
One two three four five six seven eight nine ten eleven twelve thirteen
```

text128 ncnn 结果：

```text
One two three four five six seven eight nine ten eleven twelve thirteen fourteen.
```

结论：`text_seq_len=64` 会截断这个样例；重新导出 `text_seq_len=128` 后通过。
