from pydub import AudioSegment

# 读取音频文件
audio = AudioSegment.from_wav("./Real Noise Examples/D110.wav")

# 定义分割时间点（单位为毫秒）
start_time = 5000  # 开始分割的时间
end_time = 15000  # 结束分割的时间

# 分割音频
segment = audio[start_time:end_time]

# 保存分割后的音频
segment.export("./Real Noise Examples/D110_1.1.wav", format="wav")