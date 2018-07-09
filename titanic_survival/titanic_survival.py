# Load in our libraries
import pandas as pd
import numpy as np
import re  # re >> 정규표현 연산 re(Regular expression operation)
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

train['CategoricalAge'] = pd.cut(train['Age'], 5)


# 승객 이름으로 부터 타이틀 추출(정규표현식 좀 더 공부할 것)
def get_title(name):
    # ([A - Za - z] +)\.     >>>     정규표현식  영문만 가능하다는 의미
    title_search = re.search(' ([A-Za-z]+)\.', name)

    if title_search:
        return title_search.group(1)  # list.group()  >> 정규표현식으로 매치된 문자열을 리턴

    return ""


# Title 이라는 새로운 피쳐 추가
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)

for dataset in full_data:
    # replace(이거를, 이거로 바꾼다)
    dataset['Title'] = dataset['Title'].replace(
        ['Lady', 'Couness', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Johnkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

for dataset in full_data:
    # Sex 피쳐를 매핑(?) 하여 여성은 0, 남성은 1 로 지정
    dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)

    title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
    # 위에서 지정한 매핑(?) 값으로 텍스트를 수치데이터로 치환
    dataset['Title'] = dataset['Title'].map(title_mapping)
    # Title 피처를 검사하며 NaN 값이 있으면 0으로 치환
    dataset['Title'] = dataset['Title'].fillna(0)

    # Title 값이 부동소수점 형식으로 나와서 임의로 추가
    dataset['Title'] = dataset['Title'].astype(int)

    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    # Mapping fare
    # loc 인덱서 >>  리스트.loc[행 인덱스, 열 인덱스]  >> 행, 열 인덱스에 따른 값을 찾아냄
    # 여기서는   >>  리스트.loc[행 인덱스 조건, 열 인덱스]  >> 행, 열 인덱스에 따른 값을 찾아냄
    # 왠지는 모르겠는데 여기서는 조건식에서 and 가 아니라 & 을 써야함!!!
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(7.91 < dataset['Fare']) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(14.454 < dataset['Fare']) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(16 < dataset['Age']) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(32 < dataset['Age']) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(48 < dataset['Age']) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[64 < dataset['Age'], 'Age'] = 4


# Feature selection
drop_element = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
train = train.drop(drop_element, axis=1)
train = train.drop(['CategoricalAge', 'CategoricalFare'], axis=1)

test = test.drop(drop_element, axis=1)

print(train.head(3))


