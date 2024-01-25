
#민우님은 처음2개(1,2번째)
#우석형님은 다음2개 (3,4번째)
#김우석형님 바보

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
import pandas as pd
import re
import time
import datetime

#selenium, webdriver_manager 설치

category = ['Politics', 'Economy', 'social', 'Culture', 'World', 'IT']

options = ChromeOptions()
user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
options.add_argument('user_agent=' + user_agent)
options.add_argument('lang=ko_KR')

service = ChromeService(executable_path=ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)
#pages = [164, 352, 556, 81, 107, 81] #각 카테고리 뉴스마다의 마지막 페이지. 다 긁어올거면 이렇게 하면 된다.
                                    #근데 학습할 때 데이터의 불균형이 있으면 판단을 제대로 못한다.
pages = [105, 105, 105, 81, 105, 81] #그래서 학습데이터 개수 균형 맞춰주기

df_titles = pd.DataFrame()

for l in range(2):  #정치인지, 경제인지...

    section_url = 'https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=10{}'.format(l)
    titles = []
    for k in range(1, pages[l]): #1~105페이지 긁기위해
        url = section_url +'#&date=%2000:00:00&page={}'.format(k) #페이지 마다 뒷숫자만 달라지는 주소
        try:
            driver.get(url) #여기가 브라우저 여는 부분
            time.sleep(0.5)
        except:
            print('driver.get', l, k)
        time.sleep(0.5)
        for i in range(1, 5):
            for j in range(1, 6):
                try:
                    title = driver.find_element('xpath',
                            '//*[@id="section_body"]/ul[{}]/li[{}]/dl/dt[2]/a'.format(i, j)).text
                    title = re.compile('[^가-힣]').sub(' ', title)
                    titles.append(title)
                except:
                    print('find element', l, k, i, j)
        if k % 5 == 0:
            print(l, k)
            df_section_title = pd.DataFrame(titles, columns=['titles'])
            df_section_title['category'] = category[l]
            df_section_title.to_csv('./crawling_data/data_{}_{}.csv'.format(l, k), index =False) #저장
            titles=[] #titles비워서
            #df_titles = pd.concat([df_titles, df_section_title], axis='rows', ignore_index=True)

driver.close()
#df_titles.to_csv()



