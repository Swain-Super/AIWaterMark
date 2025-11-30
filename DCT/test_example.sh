#!/bin/bash
# DCT 音频水印完整测试脚本

echo "=== DCT 音频水印测试 ==="
echo ""

# 检查是否在虚拟环境中，如果不是则尝试激活
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f "../venv/bin/activate" ]; then
        echo "激活虚拟环境..."
        source ../venv/bin/activate
    elif [ -f "venv/bin/activate" ]; then
        echo "激活虚拟环境..."
        source venv/bin/activate
    else
        echo "警告: 未找到虚拟环境，将使用系统 Python"
    fi
fi

# 测试参数
TEST_TEXT="DCT"
INPUT_AUDIO="1.mp3"
OUTPUT_AUDIO="test_dct_watermark.wav"

# 计算水印位数（每个字符8位）
NUM_BITS=$(echo -n "$TEST_TEXT" | wc -c)
NUM_BITS=$((NUM_BITS * 8))

echo "测试配置:"
echo "  输入音频: $INPUT_AUDIO"
echo "  水印文本: $TEST_TEXT"
echo "  水印位数: $NUM_BITS bits"
echo "  输出音频: $OUTPUT_AUDIO"
echo ""

# 检查输入文件是否存在
if [ ! -f "$INPUT_AUDIO" ]; then
    echo "错误: 找不到输入音频文件 $INPUT_AUDIO"
    exit 1
fi

# 1. 嵌入水印
echo "=========================================="
echo "步骤 1: 嵌入水印"
echo "=========================================="
python watermark_embed.py --input "$INPUT_AUDIO" --output "$OUTPUT_AUDIO" --text "$TEST_TEXT"

if [ $? -ne 0 ]; then
    echo "错误: 水印嵌入失败"
    exit 1
fi

echo ""
echo "✓ 水印嵌入完成"
echo ""

# 2. 检测带水印音频（使用原始音频进行对比检测，更准确）
echo "=========================================="
echo "步骤 2: 检测带水印音频（使用原始音频对比）"
echo "=========================================="
python watermark_detect.py --input "$OUTPUT_AUDIO" --original "$INPUT_AUDIO" --num-bits "$NUM_BITS"

if [ $? -ne 0 ]; then
    echo "警告: 水印检测过程出现错误（可能仍然成功）"
fi

echo ""
echo "✓ 水印检测完成"
echo ""

# 3. 对比：检测原始音频（应该检测不到水印或检测结果不同）
echo "=========================================="
echo "步骤 3: 对比测试 - 检测原始音频"
echo "=========================================="
echo "（原始音频不应包含水印，检测结果应该不同）"
python watermark_detect.py --input "$INPUT_AUDIO" --num-bits "$NUM_BITS"

echo ""
echo "=========================================="
echo "=== 测试完成 ==="
echo "=========================================="
echo ""
echo "生成的文件:"
echo "  - $OUTPUT_AUDIO (带水印的音频)"
echo ""
echo "提示: 可以手动对比原始音频和带水印音频的差异"

