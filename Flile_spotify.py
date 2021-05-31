import sqlite3
import pandas as pd
import itertools
import numpy as np

# pandasでカレントディレクトリにあるcsvファイルを読み込む
df = pd.read_csv("/Users/cocoa/Desktop/all_tracks_renew.csv")
print(df.head(5))

#すべての項目について
df.columns = ['ID','track_name','track_id', 'artist_name', 'popularity', 'acousticness', 'danceability','energy', 'liveness', 'loudness', 'valence', 'key', 'mode', 'tempo', 'release_date', 'genre']
#ジャンルの名前
genre=["acoustic","ambient","anime","edm","hip-hop","j-dance","j-idol","j-pop","j-rock","techno","trip-hop"]
#平均値
avg=['46.472727272727276','0.5901403636363641','0.2598138523','0.679566945454545', '0.1843397272727272','-7.101844545454546', '0.5276330909090907', '121.81860636363645']
#中央値
median=['50','0.12','0.595','0.7405', '0.135','-5.2265', '0.5395', '120.206']

dbname = 'TEST1.db'

conn = sqlite3.connect(dbname)
cur = conn.cursor()

# dbのnameをsampleとし、読み込んだcsvファイルをsqlに書き込む
df.to_sql('sample', conn, if_exists='replace')

#'SELECT track_name FROM sample WHERE (genre ="acoustic" and valence > 0.5276330909090907 and danceability > 0.05 and acousticness > 0.5901403636363641 and liveness >)'
# select_sql_where = 'SELECT track_name FROM sample  WHERE'+ '(popularity> 80 and danceability > 0.05)'

#質問の全選択肢を考える
hutougou=['>=','<']
bool=[]
for e in itertools.product([0, 1], repeat=3):
    bool.append(e)

print(bool)
sql=[]

#3つの質問について考えている、4つでも試してみたがイマイチ...?
for k in range(len(genre)):
    for i in range(len(bool)):
        one=hutougou[bool[i][0]]
        two=hutougou[bool[i][1]]
        three=hutougou[bool[i][2]]
        # four=hutougou[bool[i][3]]
        select_sql_where = f'SELECT track_name FROM sample WHERE (genre = "{genre[k]}" and valence'+ one +'0.5395 and danceability'+two+' 0.595 and acousticness'+three+' 0.12)'
        # select_sql_where = f'SELECT track_name FROM sample WHERE (genre = "{genre[k]}" and valence'+ one +'0.5395 and danceability'+two+' 0.595 and acousticness'+three+' 0.12 and energy'+four+' 0.7405)'
        sql.append(select_sql_where)

#全部のSQL文が、見れる
print(sql)

cnt_list=[]

#最高でも8曲選んでくるようにした
for i in range(len(sql)):
    cnt=0
    for row in cur.execute(sql[i]):
        if cnt <= 7:
            cnt+=1
            print(row)
    cnt_list.append(cnt)
    print("----")

cnt_list=np.array(cnt_list).reshape(11,8)
print(cnt_list)
cur.close()
conn.close()
