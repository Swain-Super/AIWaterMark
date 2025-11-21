# AI音频水印示例工程

## 环境准备
1. 创建虚拟环境：`python3 -m venv venv`
2. 激活虚拟环境：`source venv/bin/activate`（Windows: `venv\Scripts\activate`）
3. 安装依赖：`pip install audioseal torchaudio huggingface_hub`

**重要**：运行脚本前必须先激活虚拟环境，或使用 `venv/bin/python` 运行

## 快速生成示例音频
- `ffmpeg -f lavfi -i "sine=frequency=440:duration=5" demo.wav`
- 或运行 `python - <<'PY'\nimport numpy as np, soundfile as sf\nsr=16000\nt=np.linspace(0,5,sr*5,False)\nw=np.sin(2*np.pi*440*t).astype(np.float32)\nsf.write('demo.wav',w,sr)\nPY`

## 加密（嵌入水印）
```bash
# 方法1：激活虚拟环境后运行
source venv/bin/activate
python watermark_embed.py --input demo.wav --output demo_mark.wav --text "AB"

# 方法2：直接使用虚拟环境的 Python
venv/bin/python watermark_embed.py --input demo.wav --output demo_mark.wav --text "AB"
```

**注意**：当前模型支持 16 bits，只能编码 2 个字节
- ASCII 字符（A-Z, a-z, 0-9）：每个字符 1 字节，可编码 2 个字符，如 "AB"、"12"、"Hi"
- 中文字符：每个字符 3 字节，16 bits 只能编码部分字节，无法完整显示

## 解密（检测水印）
```bash
# 方法1：激活虚拟环境后运行
source venv/bin/activate
python watermark_detect.py --input demo_mark.wav

# 方法2：直接使用虚拟环境的 Python
venv/bin/python watermark_detect.py --input demo_mark.wav
```

## 完整测试示例
```bash
# 1. 嵌入水印（使用 ASCII 字符）
source venv/bin/activate
python watermark_embed.py --input 1.mp3 --output demo_test.wav --text "AB"

# 2. 检测水印
python watermark_detect.py --input demo_test.wav

# 预期输出：
# 检测得分: 1.0
# 水印内容: AB
# 原始二进制: [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0]

# 3. 对比原始音频（应该检测不到水印）
python watermark_detect.py --input 1.mp3
# 预期输出：检测得分接近 0（如 0.001）
```

