import numpy as np
import cv2
import math
import pygame.mixer
import pygame.midi
import pretty_midi
import random
import time
import itertools
import sqlite3
import pandas as pd


img = cv2.imread('/Users/cocoa/Desktop/oda_proj/gazou/music_zentai.png',cv2.IMREAD_UNCHANGED)
# img1 = cv2.imread('/Users/cocoa/Desktop/oda_proj/gazou/douga.png',cv2.IMREAD_UNCHANGED)
img2 = cv2.imread('/Users/cocoa/Desktop/oda_proj/gazou/disc.png',cv2.IMREAD_UNCHANGED)
img3 = cv2.imread('/Users/cocoa/Desktop/text_start.png',-1)

#ボタンの配置の座標
lis=[(130,340),(715,340),(195,150),(652,150),(425,50)]
color_list=[(255,0,0),(0,0,255),(0,255,0),(0,255,255),(255,0,102)]
note_name=['C5','D5','E5','F5','G5','A5','B5','C6']
result = []

for n in range(1,4):
    for conb in itertools.combinations(note_name, n):
        result.append(list(conb)) #タプルをリスト型に変換
note_name=result

def get_distance(x1, y1, x2, y2):
    d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return d

# def gazou_syori(frame,img):
#     bgb, bgg, bgr = cv2.split(frame)

#     fgb, fgg, fgr, fga = cv2.split(img)
#     rows, cols, ch = frame.shape

#     warpb = np.zeros((rows, cols), np.uint8)
#     warpg = np.zeros((rows, cols), np.uint8)
#     warpr = np.zeros((rows, cols), np.uint8)
#     warpa = np.zeros((rows, cols), np.uint8)

#     mat = np.array([[1.0, 0.0, 200.0], [0.0, 1.0, 200.0]], dtype=np.float32)

#     cv2.warpAffine(fgb, mat, (cols, rows), warpb, borderMode=cv2.BORDER_TRANSPARENT)
#     cv2.warpAffine(fgg, mat, (cols, rows), warpg, borderMode=cv2.BORDER_TRANSPARENT)
#     cv2.warpAffine(fgr, mat, (cols, rows), warpr, borderMode=cv2.BORDER_TRANSPARENT)
#     cv2.warpAffine(fga, mat, (cols, rows), warpa, borderMode=cv2.BORDER_TRANSPARENT)

#     bgb = bgb / 255.0
#     bgg = bgg / 255.0
#     bgr = bgr / 255.0

#     warpb = warpb / 255.0
#     warpg = warpg / 255.0
#     warpr = warpr / 255.0
#     warpa = warpa / 255.0

#     bgb = (1.0 - warpa) * bgb + warpa * warpb
#     bgg = (1.0 - warpa) * bgg + warpa * warpg
#     bgr = (1.0 - warpa) * bgr + warpa * warpr

#     result = cv2.merge((bgb, bgg, bgr))
#     return result

# def gazou_syori_cnt(frame,img,img1,cnt):
#     bgb, bgg, bgr = cv2.split(frame)

#     fgb, fgg, fgr, fga = cv2.split(img)
#     fgb1, fgg1, fgr1, fga1 = cv2.split(img1)
#     rows, cols, ch = frame.shape

#     warpb = np.zeros((rows, cols), np.uint8)
#     warpg = np.zeros((rows, cols), np.uint8)
#     warpr = np.zeros((rows, cols), np.uint8)
#     warpa = np.zeros((rows, cols), np.uint8)

#     warpb1 = np.zeros((rows, cols), np.uint8)
#     warpg1 = np.zeros((rows, cols), np.uint8)
#     warpr1 = np.zeros((rows, cols), np.uint8)
#     warpa1 = np.zeros((rows, cols), np.uint8)

#     mat = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
#     mat1 = np.array([[0.3, 0.0, 2.0*cnt+60], [0.0, 0.3, 380]], dtype=np.float32)

#     cv2.warpAffine(fgb, mat, (cols, rows), warpb, borderMode=cv2.BORDER_TRANSPARENT)
#     cv2.warpAffine(fgg, mat, (cols, rows), warpg, borderMode=cv2.BORDER_TRANSPARENT)
#     cv2.warpAffine(fgr, mat, (cols, rows), warpr, borderMode=cv2.BORDER_TRANSPARENT)
#     cv2.warpAffine(fga, mat, (cols, rows), warpa, borderMode=cv2.BORDER_TRANSPARENT)

#     cv2.warpAffine(fgb1, mat1, (cols, rows), warpb1, borderMode=cv2.BORDER_TRANSPARENT)
#     cv2.warpAffine(fgg1, mat1, (cols, rows), warpg1, borderMode=cv2.BORDER_TRANSPARENT)
#     cv2.warpAffine(fgr1, mat1, (cols, rows), warpr1, borderMode=cv2.BORDER_TRANSPARENT)
#     cv2.warpAffine(fga1, mat1, (cols, rows), warpa1, borderMode=cv2.BORDER_TRANSPARENT)

#     bgb = bgb / 255.0
#     bgg = bgg / 255.0
#     bgr = bgr / 255.0

#     warpb = warpb / 255.0
#     warpg = warpg / 255.0
#     warpr = warpr / 255.0
#     warpa = warpa / 255.0

#     warpb1 = warpb1 / 255.0
#     warpg1 = warpg1 / 255.0
#     warpr1 = warpr1 / 255.0
#     warpa1 = warpa1 / 255.0

#     bgb = (1.0 - warpa) * bgb + warpa * warpb
#     bgg = (1.0 - warpa) * bgg + warpa * warpg
#     bgr = (1.0 - warpa) * bgr + warpa * warpr

#     bgb1 = (1.0 - warpa1) * bgb + warpa1 * warpb1
#     bgg1 = (1.0 - warpa1) * bgg + warpa1 * warpg1
#     bgr1 = (1.0 - warpa1) * bgr + warpa1 * warpr1

#     result = cv2.merge((bgb, bgg, bgr))
#     result1 = cv2.merge((bgb1, bgg1, bgr1))
#     return result,result1

# 内蔵カメラを起動
cap = cv2.VideoCapture(0)

# OpenCVに用意されている顔認識するためのxmlファイルのパス
cascade_path = "/Users/cocoa/opencvEnv/lib/python3.7/site-packages/cv2/data/haarcascade_eye.xml"
# カスケード分類器の特徴量を取得する
cascade = cv2.CascadeClassifier(cascade_path)

# 顔に表示される枠の色を指定（白色）
color = (255,255,255)
# pygame.mixer.init()
# pygame.midi.init()
# player = pygame.midi.Output(3)

# c_chord = pretty_midi.PrettyMIDI(initial_tempo=10)
font = cv2.FONT_HERSHEY_SIMPLEX
start_list=[]
start_list2=[]
keturon_list=[]
distance=[210,260,650,260]

def start(str):
    flag1=True
    time_flag=True
    time_flag2=True
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        
        # モノクロで表示する
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray=cv2.equalizeHist(gray)
        #顔認識の実行
        facerect = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=2, minSize=(10,10))
        if len(facerect) > 0:
            for rect in facerect:
                cv2.rectangle(frame, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), color, thickness=2)
                xx=int((tuple(rect[0:2])[0]+tuple(rect[0:2]+rect[2:4])[0])/2)
                yy=int((tuple(rect[0:2])[1]+tuple(rect[0:2]+rect[2:4])[1])/2)

                yesd = get_distance(distance[0],distance[1], xx, yy)
                nod = get_distance(distance[2],distance[3], xx, yy)
                if yesd < 50:
                    if time_flag == True:
                        start=time.time()
                        start_list.append(start)
                        time_flag=False
                else:
                    if time_flag == False:
                        end=time.time()
                        start=start_list[-1]
                        print(end-start)
                        if end-start > 2:
                            keturon_list.append(0)
                            print("complete")
                            flag1=False
                        else:
                            time_flag = True
                
                if nod < 50:
                    if time_flag2 == True:
                        start=time.time()
                        start_list2.append(start)
                        time_flag2=False
                else:
                    if time_flag2 == False:
                        end=time.time()
                        start=start_list2[-1]
                        print(end-start)
                        if end-start > 2:
                            keturon_list.append(1)
                            print("complete")
                            flag1=False
                        else:
                            time_flag2 = True

        frame=cv2.circle(frame,(distance[0],distance[1]),50,(0,0,255))
        frame=cv2.circle(frame,(distance[2],distance[3]),50,(255,0,0))
        frame = cv2.rectangle(frame,(0,60),(410,470),(0,0,255),3)
        frame = cv2.rectangle(frame,(430,60),(848,470),(255,0,0),3)
        frame = cv2.putText(frame,str,(10,50), font,2,(255,255,255),2,cv2.LINE_AA)
        frame = cv2.putText(frame,'Yes',(100,305), font, 4,(0,0,255),2,cv2.LINE_AA)
        frame = cv2.putText(frame,'No',(580,305), font, 4,(255,0,0),2,cv2.LINE_AA)
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & flag1==False:
            break

#質問で得られた回答を元に曲名を表示
#cv2が日本語フォントに対応していなく、表示が？？？になる

def kyoku_hyouji():
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray=cv2.equalizeHist(gray)

        facerect = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=2, minSize=(10,10))
        if len(facerect) > 0:
            for rect in facerect:
                cv2.rectangle(frame, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), color, thickness=2)
                xx=int((tuple(rect[0:2])[0]+tuple(rect[0:2]+rect[2:4])[0])/2)
                yy=int((tuple(rect[0:2])[1]+tuple(rect[0:2]+rect[2:4])[1])/2)

        for i in range(len(song_list)):
            frame = cv2.putText(frame,str(song_list[i]),(100,50*i), font, 1 ,(255,255,255), 2 ,cv2.LINE_AA)

        cv2.imshow("frame", frame)
        # cv2.imshow("frame", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

start('Do you like bright songs?')
print(keturon_list)

start('Do you like to dance?')
print(keturon_list)

start('Do you like acoustic songs?')
print(keturon_list)

df = pd.read_csv("/Users/cocoa/Desktop/all_tracks_renew.csv")

# カラム名（列ラベル）を作成。csv file内にcolumn名がある場合は、下記は不要
# pandasが自動で1行目をカラム名として認識してくれる。
df.columns = ['ID','track_name','track_id', 'artist_name', 'popularity', 'acousticness', 'danceability','energy', 'liveness', 'loudness', 'valence', 'key', 'mode', 'tempo', 'release_date', 'genre']
genre=["acoustic","ambient","anime","edm","hip-hop","j-dance","j-idol","j-pop","j-rock","techno","trip-hop"]
avg=['46.472727272727276','0.5901403636363641','0.2598138523','0.679566945454545', '0.1843397272727272','-7.101844545454546', '0.5276330909090907', '121.81860636363645']
median=['50','0.12','0.595','0.7405', '0.135','-5.2265', '0.5395', '120.206']

dbname = 'TEST1.db'

conn = sqlite3.connect(dbname)
cur = conn.cursor()

# dbのnameをsampleとし、読み込んだcsvファイルをsqlに書き込む
# if_existsで、もしすでにexpenseが存在していたら、書き換えるように指示
df.to_sql('sample', conn, if_exists='replace')

hutougou=['>=','<']

one=hutougou[keturon_list[0]]
two=hutougou[keturon_list[1]]
three=hutougou[keturon_list[2]]

# four=hutougou[bool[i][3]]
select_sql_where = 'SELECT track_name FROM sample WHERE (genre = "j-pop" and valence'+ one +'0.5395 and danceability'+two+' 0.595 and acousticness'+three+' 0.12)'
# select_sql_where = f'SELECT track_name FROM sample WHERE (genre = "{genre[k]}" and valence'+ one +'0.5395 and danceability'+two+' 0.595 and acousticness'+three+' 0.12 and energy'+four+' 0.7405)'

song_list=[]
for row in cur.execute(select_sql_where):
    song_list.append(row)
    print(row)

cur.close()
conn.close()

kyoku_hyouji()
print(song_list)