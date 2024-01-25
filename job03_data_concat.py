import pandas as pd
import glob
import datetime


data_path = glob.glob('./crawling_data/*')
print(data_path)

df = pd.DataFrame()
for path in data_path:
    df_temp = pd.read_csv(path)
    df_temp.dropna(inplace=True)
    df = pd.concat([df, df_temp])


# head라인 뉴스에서는 인덱스넘버 뺐는데 일반뉴스에서는 인덱스 안빼서, 이상해진부분.. 처리할 때는 이렇게
# for path in data_path[:-1]:
#     df_temp = pd.read_csv(path, index_col=0) #파일생성 때 인덱스 있게 되서, index_col=0으로 없에기
#     df_temp.dropna(inplace=True)
#     df = pd.concat([df, df_temp])
#
# df_temp = pd.read_csv(data_path[-1])
# df = pd.concat([df, df_temp])


print(df.head())
print(df['category'].value_counts())
df.info()
df.to_csv('./naver_news_titles_{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d')), index=False)