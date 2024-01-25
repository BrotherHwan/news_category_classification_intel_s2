import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle

df = pd.read_csv('./naver_news_titles_20240125.csv')
print(df.head())
df.info()

#모든 어절(띄어쓰기 안 된 하나의 묶음)을 숫자로 표현하는게 토큰화.
#산행이죠. 산행이다. 산행일까. 이런식으로 가지수가 너무 많으니 토큰이 너무 많아짐. 그래서 '산행'만 자른다. 이걸 형태소라고 한다.
#형태소 단위로 잘라줘야한다.
#okt = Okt() #형태소로 잘라주는 작업해주는게 OKt

X = df['titles']
Y = df['category']

label_encoder = LabelEncoder()
labeled_y = label_encoder.fit_transform(Y) #Y값에 라벨에 번호 붙여주기. 이 순간 label에 정보가 있는것
print(labeled_y[:3])
label = label_encoder.classes_  #label을 어떤 번호로 붙여줬는지 정보를 가지고 있는게 classes_
                                #이건 순서가 차례로 되진 않는데 정치가 0이 아닌 다른번호에 배치된 모습
print(label)

with open('./models/label_encoder.pickle', 'wb') as f:
    pickle.dump(label_encoder, f) #파일을 f로 열어서 파일에 label_encoder 저장.
                                  #pickle은 파이썬 기본문법. 파이썬 데이터형을 그대로 저장. 원래는 문자열로 바꿔저장했다가 불러올때 바꾸고 막 이러는데
                                  #이건 float이면 float, 리스트면리스트 그대로 저장. 불러올때도 그대로 다시 불러옴.

#onehotencoding
onehot_y = to_categorical(labeled_y) #원핫인코딩 해주는 부분
print(onehot_y[:3])

#자연어처리
print(X[1:5])
okt = Okt() #형태소 분리가 아주 완벽하진 않다. '등장한->등장 한'이 되는데 '한'의 의미를 알 기 어렵다. 감탄사, 한글자 단어도 마찬가지
            #이런걸 불용어라고 하고 없에줘야한다.

for i in range(len(X)):
    X[i] = okt.morphs(X[i], stem=True) #stem을 주면 원형으로 바꿔준다. ex)공정한->공정하다
    if i % 1000:
        print(i)
#print(X[:5])

stopwords = pd.read_csv('./stopwords.csv', index_col=0)
for j in range(len(X)): #이중 인덱스로 되어 있어서 이중 for문. j는 한 문장 접근
    words = []
    for i in range(len(X[j])):
        if len(X[j][i]) > 1: #한글자 짜리 걸러내기 위해서
            if X[j][i] not in list(stopwords['stopword']):  #j는 문장접근, i는 형태소접근
                words.append(X[j][i])
    X[j] = ' '.join(words)
#print(X[:5])

token = Tokenizer()
token.fit_on_texts(X) #X안에있는 형태소에 라벨 붙여서 토큰이 그정보 가지고 있는다
tokened_x = token.texts_to_sequences(X) #토큰들을 숫자들의 리스트로 만들어주는 부분. #0은 없다. 1부터 씀
wordsize = len(token.word_index) +1 #+1한건 나중에 0을 쓰기 위해.
#print(tokened_x)
print(wordsize)

#또 다른날 긁어올 때 토크나이즈 하면 토큰 숫자가 바껴버리니까 파일로 저장해주고 그걸 그대로 쓴다.
with open('./models/news_token.pickle', 'wb') as f:
    pickle.dump(token, f)

#모든 입력문장의 길이를 맞춰주기 위해. 제일 긴문장 찾고, 그거보다 짧은건 0을 붙여줘서 사이즈 맞추기
#제일 긴문장 찾기
max = 0
for i in range(len(tokened_x)):
    if max < len(tokened_x[i]):
        max = len(tokened_x[i])
print(max)

x_pad = pad_sequences(tokened_x, max) #앞에다 알아서 0채워주는 함수
print(x_pad)

#다른날 뉴스에 정보가 없는 형태소가 나오면 그건 그냥 0 부여해준다

X_train, X_test, Y_train, Y_test = train_test_split(x_pad, onehot_y, test_size=0.2)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

xy = X_train, X_test, Y_train, Y_test #xy가 튜플이 됨.
xy = np.array(xy, dtype=object) #이거 안하면 저장이 안되서 해주는 부분
np.save('./news_data_max_{}_wordsize_{}'.format(max, wordsize), xy)