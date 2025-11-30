#!/usr/bin/env python3
"""
DCT（离散余弦变换）音频水印检测脚本

从音频中检测和提取 DCT 水印信息
"""
import argparse
import numpy as np
import torch
import torchaudio
import soundfile as sf
from scipy.fftpack import dct
from typing import List, Optional, Tuple


def bits_to_text(bits: List[int]) -> Optional[str]:
    """
    将二进制位序列转换为文本
    
    Args:
        bits: 二进制位列表
        
    Returns:
        解码后的文本，如果解码失败返回 None
    """
    if len(bits) % 8 != 0:
        return None
    
    text_bytes = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            if i + j < len(bits):
                byte |= (bits[i + j] << (7 - j))
        text_bytes.append(byte)
    
    try:
        return text_bytes.decode('utf-8')
    except UnicodeDecodeError:
        return None


def detect_watermark_dct(
    audio_data: np.ndarray,
    num_bits: int,
    block_size: int = 1024,
    start_freq: int = 50,
    end_freq: int = 200
) -> Tuple[List[int], float]:
    """
    从音频中检测 DCT 水印
    
    Args:
        audio_data: 音频数据（一维数组）
        num_bits: 期望的水印位数
        block_size: DCT 块大小（应与嵌入时一致）
        start_freq: 水印嵌入起始频率索引（应与嵌入时一致）
        end_freq: 水印嵌入结束频率索引（应与嵌入时一致）
        
    Returns:
        (检测到的水印位序列, 平均检测置信度)
    """
    audio_len = len(audio_data)
    num_blocks = (audio_len + block_size - 1) // block_size
    
    detected_bits = []
    confidence_scores = []
    
    # 对每个块进行检测
    for i in range(min(num_blocks, num_bits)):
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
        
        # 选择用于检测的频率索引（与嵌入时相同）
        freq_indices = np.linspace(
            start_freq,
            min(end_freq, len(dct_coeffs) - 1),
            num=min(10, len(dct_coeffs) - start_freq),
            dtype=int
        )
        
        # 计算这些频率系数的平均值
        coeff_values = []
        for freq_idx in freq_indices:
            if freq_idx < len(dct_coeffs):
                coeff_values.append(dct_coeffs[freq_idx])
        
        if len(coeff_values) > 0:
            avg_coeff = np.mean(coeff_values)
            # 如果平均系数为正，可能是 bit=1；如果为负，可能是 bit=0
            # 但这种方法比较粗糙，实际应该与原始音频对比
            
            # 简化检测：基于系数的符号和大小
            # 这里使用一个简单的启发式方法
            # 实际应用中，可能需要原始音频进行对比检测
            
            # 检测位：如果系数整体偏正，可能是 1；偏负可能是 0
            detected_bit = 1 if avg_coeff > 0 else 0
            confidence = abs(avg_coeff) / (np.std(coeff_values) + 1e-10)
            
            detected_bits.append(detected_bit)
            confidence_scores.append(confidence)
    
    avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
    
    return detected_bits, avg_confidence


def detect_watermark_with_original(
    watermarked_audio: np.ndarray,
    original_audio: np.ndarray,
    num_bits: int,
    block_size: int = 1024,
    start_freq: int = 50,
    end_freq: int = 200
) -> Tuple[List[int], float]:
    """
    使用原始音频进行对比检测（更准确的方法）
    
    Args:
        watermarked_audio: 带水印的音频数据
        original_audio: 原始音频数据
        num_bits: 期望的水印位数
        block_size: DCT 块大小
        start_freq: 水印嵌入起始频率索引
        end_freq: 水印嵌入结束频率索引
        
    Returns:
        (检测到的水印位序列, 平均检测置信度)
    """
    min_len = min(len(watermarked_audio), len(original_audio))
    watermarked_audio = watermarked_audio[:min_len]
    original_audio = original_audio[:min_len]
    
    num_blocks = (min_len + block_size - 1) // block_size
    detected_bits = []
    confidence_scores = []
    
    for i in range(min(num_blocks, num_bits)):
        start_idx = i * block_size
        end_idx = min(start_idx + block_size, min_len)
        
        # 处理带水印音频块
        wm_block = watermarked_audio[start_idx:end_idx]
        if len(wm_block) < block_size:
            padded_wm = np.zeros(block_size)
            padded_wm[:len(wm_block)] = wm_block
            wm_block = padded_wm
        
        # 处理原始音频块
        orig_block = original_audio[start_idx:end_idx]
        if len(orig_block) < block_size:
            padded_orig = np.zeros(block_size)
            padded_orig[:len(orig_block)] = orig_block
            orig_block = padded_orig
        
        # DCT 变换
        wm_dct = dct(wm_block, norm='ortho')
        orig_dct = dct(orig_block, norm='ortho')
        
        # 计算差异
        freq_indices = np.linspace(
            start_freq,
            min(end_freq, len(wm_dct) - 1),
            num=min(10, len(wm_dct) - start_freq),
            dtype=int
        )
        
        diff_sum = 0.0
        orig_sum = 0.0
        for freq_idx in freq_indices:
            if freq_idx < len(wm_dct):
                diff = wm_dct[freq_idx] - orig_dct[freq_idx]
                diff_sum += diff
                orig_sum += abs(orig_dct[freq_idx])
        
        # 如果差异为正，可能是 bit=1；如果为负，可能是 bit=0
        if orig_sum > 1e-10:
            normalized_diff = diff_sum / orig_sum
            detected_bit = 1 if normalized_diff > 0 else 0
            confidence = abs(normalized_diff)
        else:
            detected_bit = 0
            confidence = 0.0
        
        detected_bits.append(detected_bit)
        confidence_scores.append(confidence)
    
    avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
    return detected_bits, avg_confidence


def detect_audio(input_audio: str, original_audio: Optional[str] = None,
                 num_bits: Optional[int] = None, block_size: int = 1024) -> None:
    """
    检测音频中的 DCT 水印
    
    Args:
        input_audio: 待检测的音频文件路径
        original_audio: 原始音频文件路径（可选，如果提供会更准确）
        num_bits: 期望的水印位数（可选）
        block_size: DCT 块大小（应与嵌入时一致）
    """
    # 加载待检测音频
    try:
        waveform, sample_rate = torchaudio.load(input_audio)
    except RuntimeError:
        audio_data, sample_rate = sf.read(input_audio)
        if audio_data.ndim > 1:
            audio_data = audio_data[:, 0]
        waveform = torch.from_numpy(audio_data).unsqueeze(0).float()
    
    if isinstance(waveform, torch.Tensor):
        audio_array = waveform.squeeze(0).numpy()
        if audio_array.ndim > 1:
            audio_array = audio_array[0]
    else:
        audio_array = waveform
    
    audio_array = audio_array.astype(np.float64)
    
    # 加载原始音频（如果提供）
    original_array = None
    if original_audio:
        try:
            orig_waveform, _ = torchaudio.load(original_audio)
        except RuntimeError:
            orig_data, _ = sf.read(original_audio)
            if orig_data.ndim > 1:
                orig_data = orig_data[:, 0]
            orig_waveform = torch.from_numpy(orig_data).unsqueeze(0).float()
        
        if isinstance(orig_waveform, torch.Tensor):
            original_array = orig_waveform.squeeze(0).numpy()
            if original_array.ndim > 1:
                original_array = original_array[0]
        else:
            original_array = orig_waveform
        original_array = original_array.astype(np.float64)
    
    # 如果没有指定位数，尝试检测常见的位数（8的倍数）
    if num_bits is None:
        # 估算可能的位数（基于音频长度和块大小）
        estimated_bits = len(audio_array) // block_size
        # 尝试常见的位数：8, 16, 24, 32, 64, 128
        possible_bits = [8, 16, 24, 32, 64, 128, 256]
        num_bits = min([b for b in possible_bits if b <= estimated_bits], default=16)
        print(f"未指定水印位数，尝试检测 {num_bits} 位")
    
    # 检测水印
    if original_array is not None:
        print("使用原始音频进行对比检测（更准确）...")
        detected_bits, confidence = detect_watermark_with_original(
            audio_array, original_array, num_bits, block_size
        )
    else:
        print("使用独立检测方法（需要原始音频以获得更准确的结果）...")
        detected_bits, confidence = detect_watermark_dct(
            audio_array, num_bits, block_size
        )
    
    # 尝试将位序列转换为文本
    text = bits_to_text(detected_bits)
    
    print(f"\n检测结果:")
    print(f"检测到的位数: {len(detected_bits)}")
    print(f"平均置信度: {confidence:.4f}")
    print(f"检测到的位序列: {detected_bits[:32]}{'...' if len(detected_bits) > 32 else ''}")
    
    if text:
        print(f"水印内容: {text}")
    else:
        # 显示十六进制
        if len(detected_bits) % 8 == 0:
            hex_str = ' '.join([
                f'{int("".join(map(str, detected_bits[i:i+8])), 2):02x}'
                for i in range(0, len(detected_bits), 8)
            ])
            print(f"无法解码为文本，十六进制: {hex_str}")
        else:
            print("无法解码为文本（位数不是8的倍数）")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="使用 DCT 技术检测音频水印",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本检测（不需要原始音频）
  python watermark_detect.py --input watermarked.wav --num-bits 16
  
  # 使用原始音频进行对比检测（更准确）
  python watermark_detect.py --input watermarked.wav --original original.wav --num-bits 16
  
  # 自动检测位数
  python watermark_detect.py --input watermarked.wav

注意: 如果提供原始音频，检测结果会更准确
        """
    )
    parser.add_argument(
        "--input",
        required=True,
        help="待检测的音频文件路径"
    )
    parser.add_argument(
        "--original",
        type=str,
        default=None,
        help="原始音频文件路径（可选，提供后检测更准确）"
    )
    parser.add_argument(
        "--num-bits",
        type=int,
        default=None,
        help="期望的水印位数（可选，如果不提供会尝试自动检测）"
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=1024,
        help="DCT 块大小（应与嵌入时一致，默认 1024）"
    )
    
    args = parser.parse_args()
    
    if args.block_size < 64:
        parser.error("块大小必须至少为 64")
    
    detect_audio(
        args.input,
        args.original,
        args.num_bits,
        args.block_size
    )


if __name__ == "__main__":
    main()

