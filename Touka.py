import numpy as np
import cv2
img = cv2.imread('/Users/cocoa/Desktop/oda_proj/gazou/music_zentai.png',cv2.IMREAD_UNCHANGED)
img1 = cv2.imread('/Users/cocoa/Desktop/oda_proj/gazou/douga.png',cv2.IMREAD_UNCHANGED)
img2 = cv2.imread('/Users/cocoa/Desktop/oda_proj/gazou/disc.png',cv2.IMREAD_UNCHANGED)
cnt=0

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
    mat1 = np.array([[0.3, 0.0, 2.0*cnt+60], [0.0, 0.3, 395.0]], dtype=np.float32)

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
        ret, frame = cap.read()
        
        # モノクロで表示する
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray=cv2.equalizeHist(gray)

        # 顔認識の実行
        facerect = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=2, minSize=(10,10))
        # 顔が見つかったらcv2.rectangleで顔に白枠を表示する
        if len(facerect) > 0:
            for rect in facerect:
                cv2.rectangle(frame, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), color, thickness=2)
        
        result,result1=gazou_syori_cnt(frame,img,img1,cnt)
        cnt+=1
        # 表示
        cv2.imshow("frame", result)
        cv2.imshow("frame", result1)

        # qキーを押すとループ終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 内蔵カメラを終了
    cap.release()
    cv2.destroyAllWindows()
