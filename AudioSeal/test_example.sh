#!/bin/bash
# 音频水印完整测试示例

echo "=== 音频水印测试示例 ==="
echo ""

# 激活虚拟环境
source venv/bin/activate

# 测试文本（ASCII 字符，16 bits 可完整编码 2 个字符）
TEST_TEXT="AB"
INPUT_AUDIO="1.mp3"
OUTPUT_AUDIO="test_watermark.wav"

echo "1. 嵌入水印: $TEST_TEXT"
python watermark_embed.py --input "$INPUT_AUDIO" --output "$OUTPUT_AUDIO" --text "$TEST_TEXT"

echo ""
echo "2. 检测带水印音频"
python watermark_detect.py --input "$OUTPUT_AUDIO"

echo ""
echo "3. 对比：检测原始音频（应该检测不到水印）"
python watermark_detect.py --input "$INPUT_AUDIO"

echo ""
echo "=== 测试完成 ==="

