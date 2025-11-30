#!/usr/bin/env python3
"""
DCT（离散余弦变换）音频水印嵌入脚本

基于 DCT 变换的音频水印技术：
1. 将音频信号分块
2. 对每个块进行 DCT 变换到频域
3. 在 DCT 系数中嵌入水印信息（通常在中频系数中）
4. 进行逆 DCT 变换回时域
5. 重组音频块

优点：水印嵌入在频域，对音频质量影响小，具有一定的鲁棒性
"""
import argparse
import numpy as np
import torch
import torchaudio
import soundfile as sf
from scipy.fftpack import dct, idct
from typing import List


def text_to_bits(text: str) -> List[int]:
    """
    将文本转换为二进制位序列
    
    Args:
        text: 输入文本
        
    Returns:
        二进制位列表
    """
    bits = []
    text_bytes = text.encode('utf-8')
    for byte in text_bytes:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def embed_watermark_dct(
    audio_data: np.ndarray,
    watermark_bits: List[int],
    block_size: int = 1024,
    alpha: float = 0.1,
    start_freq: int = 50,
    end_freq: int = 200
) -> np.ndarray:
    """
    使用 DCT 变换在音频中嵌入水印
    
    Args:
        audio_data: 音频数据（一维数组）
        watermark_bits: 水印位序列
        block_size: DCT 块大小（默认 1024）
        alpha: 水印强度（默认 0.1，越大水印越明显但可能影响音质）
        start_freq: 水印嵌入起始频率索引（默认 50）
        end_freq: 水印嵌入结束频率索引（默认 200）
        
    Returns:
        嵌入水印后的音频数据
    """
    audio_len = len(audio_data)
    num_blocks = (audio_len + block_size - 1) // block_size
    num_bits = len(watermark_bits)
    
    # 确保有足够的块来嵌入所有水印位
    if num_blocks < num_bits:
        raise ValueError(
            f"音频太短，无法嵌入 {num_bits} 位水印。"
            f"需要至少 {num_bits} 个块（每个块 {block_size} 个采样点），"
            f"但只有 {num_blocks} 个块。"
        )
    
    watermarked_audio = np.zeros_like(audio_data)
    watermark_index = 0
    
    # 对每个块进行处理
    for i in range(num_blocks):
        start_idx = i * block_size
        end_idx = min(start_idx + block_size, audio_len)
        block = audio_data[start_idx:end_idx]
        
        # 如果块长度不足，进行零填充
        if len(block) < block_size:
            padded_block = np.zeros(block_size)
            padded_block[:len(block)] = block
            block = padded_block
        
        # 对块进行 DCT 变换
        dct_coeffs = dct(block, norm='ortho')
        
        # 如果还有水印位需要嵌入，在当前块中嵌入
        if watermark_index < num_bits:
            bit = watermark_bits[watermark_index]
            
            # 选择中频系数进行嵌入（避免低频和高频）
            # 使用多个系数来增强鲁棒性
            freq_indices = np.linspace(
                start_freq,
                min(end_freq, len(dct_coeffs) - 1),
                num=min(10, len(dct_coeffs) - start_freq),
                dtype=int
            )
            
            # 嵌入水印：如果 bit=1，增加系数；如果 bit=0，减少系数
            for freq_idx in freq_indices:
                if freq_idx < len(dct_coeffs):
                    if bit == 1:
                        dct_coeffs[freq_idx] += alpha * abs(dct_coeffs[freq_idx])
                    else:
                        dct_coeffs[freq_idx] -= alpha * abs(dct_coeffs[freq_idx])
            
            watermark_index += 1
        
        # 逆 DCT 变换回时域
        watermarked_block = idct(dct_coeffs, norm='ortho')
        
        # 将处理后的块放回原位置
        watermarked_audio[start_idx:end_idx] = watermarked_block[:end_idx - start_idx]
    
    return watermarked_audio


def embed_audio(input_audio: str, output_audio: str, text: str, 
                block_size: int = 1024, alpha: float = 0.1) -> None:
    """
    使用 DCT 技术嵌入音频水印
    
    Args:
        input_audio: 输入音频文件路径
        output_audio: 输出音频文件路径
        text: 水印文本内容
        block_size: DCT 块大小（默认 1024）
        alpha: 水印强度（默认 0.1）
    """
    # 加载音频文件（支持多种格式）
    try:
        waveform, sample_rate = torchaudio.load(input_audio)
    except RuntimeError:
        # 如果 torchaudio 无法加载，使用 soundfile
        audio_data, sample_rate = sf.read(input_audio)
        if audio_data.ndim > 1:
            audio_data = audio_data[:, 0]  # 转为单声道
        waveform = torch.from_numpy(audio_data).unsqueeze(0).float()
    
    # 转换为 numpy 数组
    if isinstance(waveform, torch.Tensor):
        audio_array = waveform.squeeze(0).numpy()
        if audio_array.ndim > 1:
            audio_array = audio_array[0]  # 取第一个声道
    else:
        audio_array = waveform
    
    # 确保是浮点类型
    audio_array = audio_array.astype(np.float64)
    
    # 将文本转换为二进制位
    watermark_bits = text_to_bits(text)
    print(f"水印文本: {text}")
    print(f"水印位数: {len(watermark_bits)} bits")
    print(f"音频长度: {len(audio_array)} 采样点")
    print(f"采样率: {sample_rate} Hz")
    
    # 嵌入水印
    watermarked_audio = embed_watermark_dct(
        audio_array,
        watermark_bits,
        block_size=block_size,
        alpha=alpha
    )
    
    # 归一化到 [-1, 1] 范围
    max_val = np.max(np.abs(watermarked_audio))
    if max_val > 1.0:
        watermarked_audio = watermarked_audio / max_val
    
    # 转换为 torch tensor 并保存
    watermarked_tensor = torch.from_numpy(watermarked_audio).float()
    
    # 确保形状为 (channels, time)
    if watermarked_tensor.ndim == 1:
        watermarked_tensor = watermarked_tensor.unsqueeze(0)
    
    torchaudio.save(output_audio, watermarked_tensor, sample_rate)
    print(f"水印音频已保存到: {output_audio}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="使用 DCT 技术进行音频水印嵌入",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本用法
  python watermark_embed.py --input audio.wav --output watermarked.wav --text "MyWatermark"
  
  # 自定义块大小和水印强度
  python watermark_embed.py --input audio.wav --output watermarked.wav --text "AB" --block-size 2048 --alpha 0.15

参数说明:
  --block-size: DCT 块大小，越大水印容量越小但鲁棒性可能更好（默认 1024）
  --alpha: 水印强度，越大水印越明显但可能影响音质（默认 0.1，范围建议 0.05-0.2）
        """
    )
    parser.add_argument(
        "--input",
        required=True,
        help="输入音频文件路径"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="输出音频文件路径"
    )
    parser.add_argument(
        "--text",
        required=True,
        help="水印文本内容"
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=1024,
        help="DCT 块大小（默认 1024）"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="水印强度（默认 0.1）"
    )
    
    args = parser.parse_args()
    
    # 验证参数
    if args.block_size < 64:
        parser.error("块大小必须至少为 64")
    if args.alpha <= 0 or args.alpha > 1.0:
        parser.error("水印强度必须在 (0, 1] 范围内")
    
    embed_audio(
        args.input,
        args.output,
        args.text,
        block_size=args.block_size,
        alpha=args.alpha
    )


if __name__ == "__main__":
    main()

