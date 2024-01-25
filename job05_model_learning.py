import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

X_train, X_test, Y_train, Y_test = np.load('./news_data_max_27_wordsize_11911.npy', allow_pickle=True)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

model = Sequential()
model.add(Embedding(11911, 300, input_length=27))
# 숫자만으로는 단어 학습이 안되서 필요한 부분.
# mbedding(워드사이즈, 차원축소,) 은 자연어 학습 레이어
# 형태소 개수의 차원을 가지는 의미공간...?의미공간에 벡터화를 한다. 좌표 준다는 얘기. 이 벡터값을 가지고 학습하면 의미를 학습할 수 있다.
# 연관있는 단어는 가까이 있고 없는건 멀리 있는 느낌.
# 차원이 커지면 커질수록 밀도는 작아진다(거리가 멀어진다. 데이터 개수가 같다면). 데이터가 희소해진다. 차원의 저주 라고도 한다.
# 그래서 차원을 줄여야 한다. 차원축소. 300은 300차원까지 줄이겠다는 얘기. 너무줄이면 데이터간의 관계가 망가지고, 너무 높으면 차원의 저주.
# 데이터의 최대길이가 27. 앞에서 이거보다 짧은애들은 패딩 붙여줬었다.

model.add(Conv1D(32, kernel_size=5, padding='same', activation='relu'))
# 이미지 할 때는 con2d사용했었다. 5칸짜리 필터를 한칸씩 옆으로 옮기면서 하는것. 어순보다는 주변단어와의 앞뒤관계 학습위해 넣었다
model.add(MaxPooling1D(pool_size=1))
# 1개 중에 제일 큰값. 하나마나인데 콘브레이어가 들어가면 보통 max레이어가 따라온다. 실험결과 1(안하는거)가 좋아서 이렇게 했다.
model.add(LSTM(128, activation='tanh', return_sequences=True))
# return_sequences는 셀 하나에서 값이 나올 때 하나하나 다 쌓는다는것. CNN은 나온값이 계속돌아 들어가고 그 결과값이 또 나오고 이런식이기 때문.
# 뒤의 레이어도 sequencial한 데이터를 주려면 이렇게 쌓아서 넘겨야 한다.
# 안주면 그냥 하나만 나온다.
# 맨 마지막 RNN레이어는 해주나 마나 Flatten시키기 때문에 상관없다.
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(6, activation='softmax')) #카테고리 6개니까
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
fit_hist = model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_test, Y_test))
model.save('./models/news_category_classification_model_{}.h5'.format(fit_hist.history['val_accuracy'][-1]))
#마지막에 -1 주는 이유. val_accuracy가 매 에폭마다 fit_hist에 저장되는데 제일마지막에 저장된확률로 파일명 하려고 -1
plt.plot(fit_hist.history['val_accuracy'], label='validation accuracy')
plt.plot(fit_hist.history['accuracy'], label='train accuracy')
plt.legend()
plt.show()

