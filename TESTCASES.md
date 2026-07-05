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
  --model /data/qwen3-asr-ncnn/models/qwen3_asr_0_6b_runtime_text64 \
  --audio-wav "$wav" \
  --generate-from-features \
  --language English \
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

| 样例 | 类型 | 时长 | PyTorch | ncnn | 结论 |
| --- | --- | ---: | --- | --- | --- |
| `pdx_voice` | 真实短语音 | 4.95s | `This is a test of me recording my voice.` | `This is a test of me recording my voice.` | 通过 |
| `hello_world` | 合成短语音 | 1.36s | `Hello world.` | `Hello world.` | 通过 |
| `natural_zero` | 真实短 digit | 0.64s | `Zero.` | `Zero.` | 通过 |
| `digit_five` | 真实短 digit | 0.42s | `Five.` | `Five.` | 通过 |
| `long_digit_five` | 人工重复压力样例 | 16.97s | `By by` | `By by by by by by by by by by ...` | 不作为语义正确性 benchmark |

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
ncnn:    By by by by by by by by by by ...
```

结论：这是长音频切块拼接问题，不是核心 ncnn 模块转换失败。
