import numpy as np
import cv2
import math
import pygame.mixer
import pygame.midi
import pretty_midi
import random
import time
import itertools

img = cv2.imread('/Users/cocoa/Desktop/oda_proj/gazou/music_zentai.png',cv2.IMREAD_UNCHANGED)
# img1 = cv2.imread('/Users/cocoa/Desktop/oda_proj/gazou/douga.png',cv2.IMREAD_UNCHANGED)
img2 = cv2.imread('/Users/cocoa/Desktop/oda_proj/gazou/disc.png',cv2.IMREAD_UNCHANGED)

#ボタンの配置の座標
lis=[(130,340),(715,340),(195,150),(652,150),(425,50)]
pygame.mixer.init()
pygame.midi.init()
# player = pygame.midi.Output(3)

c_chord = pretty_midi.PrettyMIDI()
note_name=['C5','D5','E5','F5','G5','A5','B5','C6']
result = []
for n in range(1,4):
    for conb in itertools.combinations(note_name, n):
        result.append(list(conb))
#1-3音にする
note_name=result
print(note_name)
cnt=0

def get_distance(x1, y1, x2, y2):
    d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return d

def gazou_syori_cnt(frame,img,img1,cnt):
    bgb, bgg, bgr = cv2.split(frame)

    fgb, fgg, fgr, fga = cv2.split(img)
    fgb1, fgg1, fgr1, fga1 = cv2.split(img1)
    rows, cols, ch = frame.shape

    warpb = np.zeros((rows, cols), np.uint8)
    warpg = np.zeros((rows, cols), np.uint8)
    warpr = np.zeros((rows, cols), np.uint8)
    warpa = np.zeros((rows, cols), np.uint8)

    warpb1 = np.zeros((rows, cols), np.uint8)
    warpg1 = np.zeros((rows, cols), np.uint8)
    warpr1 = np.zeros((rows, cols), np.uint8)
    warpa1 = np.zeros((rows, cols), np.uint8)

    mat = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    mat1 = np.array([[0.3, 0.0, 2.0*cnt+60], [0.0, 0.3, 380]], dtype=np.float32)

    cv2.warpAffine(fgb, mat, (cols, rows), warpb, borderMode=cv2.BORDER_TRANSPARENT)
    cv2.warpAffine(fgg, mat, (cols, rows), warpg, borderMode=cv2.BORDER_TRANSPARENT)
    cv2.warpAffine(fgr, mat, (cols, rows), warpr, borderMode=cv2.BORDER_TRANSPARENT)
    cv2.warpAffine(fga, mat, (cols, rows), warpa, borderMode=cv2.BORDER_TRANSPARENT)

    cv2.warpAffine(fgb1, mat1, (cols, rows), warpb1, borderMode=cv2.BORDER_TRANSPARENT)
    cv2.warpAffine(fgg1, mat1, (cols, rows), warpg1, borderMode=cv2.BORDER_TRANSPARENT)
    cv2.warpAffine(fgr1, mat1, (cols, rows), warpr1, borderMode=cv2.BORDER_TRANSPARENT)
    cv2.warpAffine(fga1, mat1, (cols, rows), warpa1, borderMode=cv2.BORDER_TRANSPARENT)

    bgb = bgb / 255.0
    bgg = bgg / 255.0
    bgr = bgr / 255.0

    warpb = warpb / 255.0
    warpg = warpg / 255.0
    warpr = warpr / 255.0
    warpa = warpa / 255.0

    warpb1 = warpb1 / 255.0
    warpg1 = warpg1 / 255.0
    warpr1 = warpr1 / 255.0
    warpa1 = warpa1 / 255.0

    bgb = (1.0 - warpa) * bgb + warpa * warpb
    bgg = (1.0 - warpa) * bgg + warpa * warpg
    bgr = (1.0 - warpa) * bgr + warpa * warpr

    bgb1 = (1.0 - warpa1) * bgb + warpa1 * warpb1
    bgg1 = (1.0 - warpa1) * bgg + warpa1 * warpg1
    bgr1 = (1.0 - warpa1) * bgr + warpa1 * warpr1

    result = cv2.merge((bgb, bgg, bgr))
    result1 = cv2.merge((bgb1, bgg1, bgr1))
    return result,result1

def  music(string,ran,start,end):
    print(string)
    program = pretty_midi.instrument_name_to_program(string)
    cello = pretty_midi.Instrument(program=program)
    # player.set_instrument(program)
    for note in note_name[ran]:
        note_number = pretty_midi.note_name_to_number(note)
        # Create a Note instance, starting at 0s and ending at .5s
        note = pretty_midi.Note(velocity=100, pitch=note_number, start=start, end=end)
        # Add it to our cello instrument
        cello.notes.append(note)
        # player.note_on(70, 120)
        # time.sleep(1)
        # player.note_off(70,120)
    c_chord.instruments.append(cello)

flag=True
if __name__ == "__main__":
    
    # 内蔵カメラを起動
    cap = cv2.VideoCapture(0)

    # OpenCVに用意されている顔認識するためのxmlファイルのパス
    cascade_path = "/Users/cocoa/opencvEnv/lib/python3.7/site-packages/cv2/data/haarcascade_eye.xml"
    # カスケード分類器の特徴量を取得する
    cascade = cv2.CascadeClassifier(cascade_path)
    
    # 顔に表示される枠の色を指定（白色）
    color = (255,255,255)
    while True:
        ret, frame = cap.read()
        #画像反転
        frame = cv2.flip(frame, 1)
        
        # モノクロで表示する
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray=cv2.equalizeHist(gray)

        # 顔認識の実行
        facerect = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=2, minSize=(10,10))
        # 顔が見つかったらcv2.rectangleで顔に白枠を表示する
        if len(facerect) > 0:
            for rect in facerect:
                cv2.rectangle(frame, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), color, thickness=2)
                xx=int((tuple(rect[0:2])[0]+tuple(rect[0:2]+rect[2:4])[0])/2)
                yy=int((tuple(rect[0:2])[1]+tuple(rect[0:2]+rect[2:4])[1])/2)

                #距離を測る
                min=1000000
                idx=0
                for i in range(len(lis)):
                    d = get_distance(lis[i][0],lis[i][1], xx, yy)
                    if min > d:
                        idx=i
                        min=d
                
                if min < 50:
                    if flag==True:
                        start=time.time()
                        flag=False
                    #どの和音にするか
                    ran=random.randint(0,len(note_name)-1)
                    #何秒鳴らすか
                    ran1=random.randint(1,4)
                    if idx ==0:
                        str='Viola'
                        start=time.time()-start
                        end=start+ran1
                        music(str,ran,start,end)
                        cv2.circle(frame,lis[0], 45, (255,0,0), -1)
                    if idx ==1:
                        str='Trumpet'
                        start=time.time()-start
                        end=start+ran1
                        music(str,ran,start,end)
                        cv2.circle(frame,lis[1], 45, (0,0,255), -1)
                    if idx ==2:
                        str='Violin'
                        start=time.time()-start
                        end=start+ran1
                        music(str,ran,start,end)
                        cv2.circle(frame,lis[2], 45, (0,255,0), -1)
                    if idx ==3:
                        str='Overdriven Guitar'
                        start=time.time()-start
                        end=start+ran1
                        music(str,ran,start,end)
                        cv2.circle(frame,lis[3], 45, (0,255,255), -1)
                    if idx ==4:
                        str='Electric Grand Piano'
                        start=time.time()-start
                        end=start+ran1
                        music(str,ran,start,end)
                        cv2.circle(frame,lis[4], 45, (255,0,102), -1)
                    # pygame.mixer.music.load("/Users/cocoa/Desktop/oda_proj/music/manuke.mp3")
                    # pygame.mixer.music.play(1)
        cnt+=1
        result,result1=gazou_syori_cnt(frame,img,img2,cnt)
        # 表示
        cv2.imshow("frame", frame)
        cv2.imshow("frame", result)

        # qキーを押すとループ終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            c_chord.write('Viola-C.mid')
            break

    # 内蔵カメラを終了
    pygame.midi.quit()
    cap.release()
    cv2.destroyAllWindows()