#https://qiita.com/seigot/items/62a85f1a561bb820532a　←参考サイト
import speech_recognition as sr

AUDIO_FILE = "./otamesi_voice.wav"#音声を読み込みたいwavファイル名

# use the audio file as the audio source
r = sr.Recognizer()
with sr.AudioFile(AUDIO_FILE) as source:
    audio = r.record(source)  # read the entire audio file

result=r.recognize_google(audio, language='ja-JP')

try:
    print("Google Speech Recognition thinks you said " + result)
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))