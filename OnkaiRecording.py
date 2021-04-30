# coding: utf-8
import random
import cv2
import pygame.mixer
import pyaudio
#import numpy as np
import wave
import time
from time import perf_counter
import math
#import perf_counter

start_time = 0.0
timediff = 0.0

#音の連続を避けるために無音の時間を入れる
pygame.mixer.init()
pygame.init()

muon = pygame.mixer.Sound("/Users/.../music/music_muon.mp3")

array=[]
for i in range(9):
    array.append(int(58.75*i))



def Onkai():
    global start_time
    global timediff
    start_time = perf_counter()

    # 内蔵カメラを起動
    cap = cv2.VideoCapture(0)

    # OpenCVに用意されている顔認識するためのxmlファイルのパス
    cascade_path = "/Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages/cv2/data/haarcascade_eye.xml"
    # カスケード分類器の特徴量を取得する
    cascade = cv2.CascadeClassifier(cascade_path)

    # 顔に表示される枠の色を指定
    color = (255,0,255)


    while timediff<3:


        # 内蔵カメラから読み込んだキャプチャデータを取得
        ret, frame = cap.read()

        # 顔認識の実行
        facerect = cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=2, minSize=(10,10))

        #文字を表示
        cv2.putText(frame,"Recording",(500,80), cv2.FONT_HERSHEY_SIMPLEX,3,(100,2,92),5,cv2.LINE_AA)
        # qキーを押すとループ終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break


        for i in range(9):
            red=random.randint(0,255)
            green=random.randint(0,255)
            blue=random.randint(0,255)
            cv2.line(frame,(0,array[i]+150),(1500,array[i]+150),(blue,green,red),20)

        # 顔が見つかったらcv2.rectangleで顔に枠を表示する
        if len(facerect) > 0:
            for rect in facerect:
                cv2.rectangle(frame, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), color, thickness=5)

                yy=int((tuple(rect[0:2])[1]+tuple(rect[0:2]+rect[2:4])[1])/2)

                for i in range(7):
                    if array[i]<=yy<=array[i+1]:
                        music = pygame.mixer.Sound("/Users/.../music/piano"+str(i)+".mp3")

                #pygame.mixer.music.play(1)
                        music.play()
                        muon.play()
                stopwatch()

        # 表示
        cv2.imshow("frame", frame)
    # 内蔵カメラを終了
    cap.release()
    cv2.destroyAllWindows()




def stopwatch():
    global start_time
    global timediff
    now = perf_counter()
    timediff = now - start_time
    return timediff


if __name__ == "__main__":

    DEVICE_INDEX = 0
    CHUNK = 1024*2
    FORMAT = pyaudio.paInt16 # 16bit
    CHANNELS = 1             # monaural
    RATE = 44100             # sampling frequency [Hz]

    time = 10 # record time [s]
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
    Onkai()
    
    #ここで固まる
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
