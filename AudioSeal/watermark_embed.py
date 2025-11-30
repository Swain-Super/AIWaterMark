#!/usr/bin/env python3
import argparse
import torch
import torchaudio
import soundfile as sf
from audioseal import AudioSeal


def embed_audio(input_audio: str, output_audio: str, text: str) -> None:
    # 使用 AudioSeal 嵌入水印
    generator = AudioSeal.load_generator("audioseal_wm_16bits")
    
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
    
    # 将文本转换为二进制消息
    text_bytes = text.encode('utf-8')
    nbits = generator.msg_processor.nbits if generator.msg_processor else 16
    msg = torch.zeros(1, nbits, dtype=torch.int)
    for i, byte in enumerate(text_bytes[:nbits//8]):
        for j in range(8):
            if i*8 + j < nbits:
                msg[0, i*8 + j] = (byte >> (7-j)) & 1
    
    # 嵌入水印
    watermarked = generator(waveform, sample_rate=sample_rate, message=msg)
    
    # 保存时转换为 (channels, time) 格式并移除梯度
    if watermarked.ndim == 3:
        watermarked = watermarked.squeeze(0)  # (batch, channels, time) -> (channels, time)
    watermarked = watermarked.detach()  # 移除梯度信息
    torchaudio.save(output_audio, watermarked, sample_rate)


def main() -> None:
    parser = argparse.ArgumentParser(description="音频水印加密")
    parser.add_argument("--input", required=True, help="原始音频路径")
    parser.add_argument("--output", required=True, help="输出音频路径")
    parser.add_argument("--text", required=True, help="水印内容")
    args = parser.parse_args()
    embed_audio(args.input, args.output, args.text)


if __name__ == "__main__":
    main()

