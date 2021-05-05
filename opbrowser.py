#spotifyの日本の週刊ランキングのcsvをダウンロード
import datetime
from dateutil.relativedelta import relativedelta
import webbrowser
top = datetime.date(2021,3, 26)#開始日時の設定
for i in range(20):　#取得するcsvの数
  topp = top - relativedelta(days=7*i)
  bot = topp - relativedelta(days=7)
  url = 'https://spotifycharts.com/regional/jp/weekly/' + str(bot) + '--'+ str(topp) +'/download'
  webbrowser.open(url)