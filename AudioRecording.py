import pyaudio
import wave
import time

DEVICE_INDEX = 0
#↓うまくできなかったときは1024*2にしてみてください
CHUNK = 1024
FORMAT = pyaudio.paInt16 # 16bit
CHANNELS = 1             # monaural
RATE = 48000             # sampling frequency [Hz]

#5秒間録音
time = 5 # record time [s]

output_path = "./sample.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index = DEVICE_INDEX,
                frames_per_buffer=CHUNK)

print("recording ...")

frames = []

for i in range(0, int(RATE / CHUNK * time)):
    data = stream.read(CHUNK)
    frames.append(data)

print("done.")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(output_path, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()
