import wave

import librosa
import librosa.display
import matplotlib.pyplot as plt
from pydub import AudioSegment

input_file1 = './Real Noise Examples/D110_1.1.wav'
input_file2 = './Real Noise Examples/AirConditioner_1.1.wav'
# 加载音频文件
sound1 = AudioSegment.from_file(input_file1)
sound2 = AudioSegment.from_file(input_file2)

# 确保两个音频文件具有相同的采样率
sound2 = sound2.set_frame_rate(sound1.frame_rate)

# 确保两个音频文件具有相同的声道数
sound2 = sound2.set_channels(sound1.channels)

# 如果两个音频文件的长度不同，可以选择进行截断或者填充操作
if len(sound1) > len(sound2):
    sound1 = sound1[:len(sound2)]
else:
    sound2 = sound2[:len(sound1)]

# 混合音频文件
mixed_sound = sound1.overlay(sound2)

# 保存混合后的音频文件
output_file = 'merge.wav'
mixed_sound.export(output_file, format='wav')
#
# print('混合音频文件已保存为', output_file)

# merge_wav_files(input_file1, input_file2, output_file)


# 在音频叠加（合成）过程中，合成音频的频谱并不简单地等于原始音频的频谱相加。
# 实际上，音频叠加会导致原始音频的频谱进行复杂的叠加和混合，其结果不仅仅是简单的频谱相加。
# 当两个音频文件进行叠加处理时，其频谱的混合效果是由多种因素共同决定的，包括幅度叠加、相位叠加等。
# 因此，合成音频的频谱不仅包含了原始音频文件的频谱信息，还包含了叠加处理所带来的新的频谱特征。
# 在叠加过程中，如果两个音频文件的频谱成分存在重叠，则合成音频的频谱会反映出这种重叠部分的增强效果。
# 同时，叠加过程中还可能存在相位混合等现象，导致合成音频的频谱不仅仅是简单地频谱相加。

2
# # 读取音频文件
# y1, sr1 = librosa.load(output_file)
# y2, sr2 = librosa.load(input_file1)
#
# # 绘制波形图
# plt.figure(figsize=(10, 6))
# plt.subplot(2, 1, 1)
# librosa.display.waveshow(y1, sr=sr1)
# plt.title('Waveform of ' + output_file)
#
# plt.subplot(2, 1, 2)
# librosa.display.waveshow(y2, sr=sr2)
# plt.title('Waveform of ' + input_file1)
#
# plt.tight_layout()
# plt.show()

3
# input_file = wave.open('./Real Noise Examples/CopyMachine_1.1.wav', 'rb')
# output_file = wave.open('./Real Noise Examples/CopyMachine_1.2.wav', 'wb')
#
# # 设置输出文件的参数与输入文件相同
# output_file.setparams(input_file.getparams())
#
# # 读取每个帧的数据
# frames = input_file.readframes(input_file.getnframes())
#
# # 将音频数据转换为字节数组
# samples = list(frames)
#
# # 将每个采样值乘以3扩大音量并限制在合适的范围内
# new_samples = [min(max(int(sample * 3), -32768), 32767) for sample in samples]
#
# # 将处理后的音频数据转换回字节流
# output_frames = bytes([sample // 256 for sample in new_samples])
#
# # 写入输出文件
# output_file.writeframes(output_frames)
#
# # 关闭文件
# input_file.close()
# output_file.close()