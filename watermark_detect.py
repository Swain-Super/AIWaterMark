#!/usr/bin/env python3
import argparse
import torch
import torchaudio
import soundfile as sf
from audioseal import AudioSeal


def detect_audio(input_audio: str) -> None:
    # 使用 AudioSeal 解密水印
    detector = AudioSeal.load_detector("audioseal_detector_16bits")
    
    # 加载音频文件（支持多种格式）
    try:
        waveform, sample_rate = torchaudio.load(input_audio)
    except RuntimeError:
        # 如果 torchaudio 无法加载，使用 soundfile
        audio_data, sample_rate = sf.read(input_audio)
        if audio_data.ndim > 1:
            audio_data = audio_data[:, 0]  # 转为单声道
        waveform = torch.from_numpy(audio_data).unsqueeze(0).float()
    
    # 确保 waveform 形状为 (batch, channels, time)
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(1)  # (batch, time) -> (batch, channels, time)
    
    # 检测水印
    score, msg = detector.detect_watermark(waveform, sample_rate=sample_rate)
    
    # 将二进制消息转换为文本
    msg_bits = msg[0].cpu().numpy()
    text_bytes = bytearray()
    for i in range(0, len(msg_bits), 8):
        byte = 0
        for j in range(8):
            if i + j < len(msg_bits):
                byte |= (msg_bits[i + j] << (7 - j))
        text_bytes.append(byte)
    
    # 尝试解码文本
    text = None
    try:
        # 尝试完整解码
        text = text_bytes.decode('utf-8')
    except UnicodeDecodeError:
        # 如果完整解码失败，尝试逐步解码
        for length in range(len(text_bytes), 0, -1):
            try:
                text = text_bytes[:length].decode('utf-8')
                break
            except UnicodeDecodeError:
                continue
    
    if text is None or len(text) == 0:
        # 如果无法解码，显示二进制和十六进制
        hex_str = ' '.join([f'{b:02x}' for b in text_bytes])
        text = f"二进制: {msg_bits.tolist()}\n十六进制: {hex_str}"
    
    print(f"检测得分: {score}")
    print(f"水印内容: {text}")
    if isinstance(text, str) and '\n' not in text:
        print(f"原始二进制: {msg_bits.tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="音频水印解密")
    parser.add_argument("--input", required=True, help="待检测音频路径")
    args = parser.parse_args()
    detect_audio(args.input)


if __name__ == "__main__":
    main()

