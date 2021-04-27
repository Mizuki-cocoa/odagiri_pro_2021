# coding: utf-8
import cv2 
avg = None

if __name__ == "__main__":
    
    # 内蔵カメラを起動
    cap = cv2.VideoCapture(0)

    # OpenCVに用意されている顔認識するためのxmlファイルのパス
    cascade_path = "/Users/cocoa/opencvEnv/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_alt.xml"
    # カスケード分類器の特徴量を取得する
    cascade = cv2.CascadeClassifier(cascade_path)
    
    # 顔に表示される枠の色を指定（白色）
    color = (255,255,255)

    while True:
        # 内蔵カメラから読み込んだキャプチャデータを取得
        ret, frame = cap.read()

        # 結果を出力
        # cv2.imshow("Frame", frame)

        key = cv2.waitKey(30)
            # モノクロで表示する
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 顔認識の実行
        facerect = cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=2, minSize=(10,10))
        print(facerect)
        # print(facerect)
        # 顔が見つかったらcv2.rectangleで顔に白枠を表示する
        if len(facerect) > 0:
            for rect in facerect:
                # グレースケールに変換
                # print(type(rect))
                # print(rect.shape)
                # print(frame.shape)
                # gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

                # #比較用のフレームを取得する
                # if avg is None:
                #     avg = gray.copy().astype("float")
                #     continue

                # # 現在のフレームと移動平均との差を計算
                # cv2.accumulateWeighted(gray, avg, 0.5)

                # frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

                # # デルタ画像を閾値処理を行う
                # thresh = cv2.threshold(frameDelta, 3, 255, cv2.THRESH_BINARY)[1]
                # # 画像の閾値に輪郭線を入れる
                # contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # frame = cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

                leftup=tuple(rect[0:2])
                rightdown=tuple(rect[0:2]+rect[2:4])

                leftx=leftup[0]
                lefty=leftup[1]

                rightx=rightdown[0]
                righty=rightdown[1]

                new=[]

                x3=leftx

                for i in range(righty-lefty):
                    for k in range(rightx-leftx):
                        if rightx>leftx:
                            new.append(frame[leftx][lefty])
                            leftx+=1
                    lefty+=1
                    leftx=x3
                
                print(type(new))
                cv2.rectangle(frame, leftup, rightdown, color, thickness=2)
    
        # 表示
        cv2.imshow("frame", frame)

        # qキーを押すとループ終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 内蔵カメラを終了
    cap.release()
    cv2.destroyAllWindows()
