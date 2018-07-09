# Load in our libraries
import pandas as pd
import numpy as np
import re
import sklearn
import xgboost as xgb  # xgboost : 그라디언트 부스트 라이브러리
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.offline as py

py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import warnings

warnings.filterwarnings('ignore')

# Going to use these 5 base models for the stacking
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.cross_validation import KFold

# Load dataset
train = pd.read_csv('/Users/song/Documents/kaggle/titanic_survival/input/train.csv')
test = pd.read_csv('/Users/song/Documents/kaggle/titanic_survival/input/test.csv')

# Parse 'Passenger ID' in test[]
PassengerId = test['PassengerId']  # csv 파일의 'PassengerId' 칼럼을 추출

train.head(3)

full_data = [train, test]

# 이름 칼럼을 이름길이로 변경
train['Name_length'] = train['Name'].apply(len)
test['Name_length'] = test['Name'].apply(len)

# 캐빈(선실, 객식) 을 가지고 있는냐에 따라 0을 디폴트로 조건이 성립 안되면 1을 반환
# lambda : 함수를 생성할 때 사용되는 예약어로 def 와 동일하나 한 줄로 간결하게 만들 때 사용
#  >>>>   lambda a, b: a+b
train['Has_Cabin'] = train['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test['Cabin'].apply(lambda x: 0 if type(x) == float else 1)

# Sibsp(Sibling + Spouses : 형제, 자매, 배우자), Parch(parent + child), 본인까지 총 가족 수를 계산
# 계산된 값으로 새로운 피쳐 생성
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# 혼자 탑승한 경우(동승자가 없는경우) IsAlone 값을 1로 지정
# loc 인덱서 >>  리스트.loc[행 인덱스, 열 인덱스]  >> 행, 열 인덱스에 따른 값을 찾아냄
# 계산된 값으로 새로운 피쳐 생성
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# Embarked(탑승지역) 인덱스의 모든 NaN 을 다른 값(S) 으로 대체
# 리스트.fillna(값) >> 리스트 내의 NaN 을 다른 값으로 대체   fillna >> fill nan 인듯
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

# Fare(요금) 인덱스의 NaN 값을 요금의 평균 값으로 대체
# median >> 리스트의 중간값 출력     홀수 >> 가운데 값,   짝수 >> 가운데 2개의 평균 값
# list = [1, 3, 5] >> 3      list = [1, 3, 5, 7] >> 4(3+5/2)
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

# train 리스트의 'Fare' 인덱스의 값을 4개의 등급으로 나눔
# pandas.qcut(x, n)  >> x의 값을 기준으로 n 개의 링크로 나누는 듯 ??
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)

for dataset in full_data:
    age_avg = dataset['Age'].mean()  # 평균 계산
    age_std = dataset['Age'].std()  # 표준편차 계산

    # NaN 값을 카운트
    # 리스트.isnull()  >>>  Detect missing values(NaN in numeric array, None/NaN in object arrays)
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

# 여기 무슨 말인지 모르겠다 진짜 모르겠다
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)

train['CategoricalFare'] = pd.cut(train['Age'], 5)

print(dataset)

