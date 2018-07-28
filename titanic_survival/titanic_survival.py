# Load in our libraries
import pandas as pd
import numpy as np
import re  # re >> 정규표현 연산 re(Regular expression operation)
import sklearn
import xgboost as xgb  # xgboost : 그라디언트 부스트 라이브러리
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
# 배열.drop() >> 특정 열을 삭제한 객체를 리턴, 디폴트인 axis=0 은 행(index) 을 삭제한다는 의미이며 axis=1  일땐 열을 삭제한다는 의미
drop_element = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
train = train.drop(drop_element, axis=1)
train = train.drop(['CategoricalAge', 'CategoricalFare'], axis=1)

test = test.drop(drop_element, axis=1)

print(train.head(5))

ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 0  # 재생산을 위한???  >> 아마 클래스 초기화 함수에서 초기화 되어 사용된 후 0으로 초기화를 위한 것 인듯  ?????
NFOLDS = 5  # set folds for out-of-fold prediction  ?? 이거 무슨말인지 모르겠슈

# 출처 : http://become-datascientist.tistory.com/entry/%EA%B5%90%EC%B0%A8%EA%B2%80%EC%A6%9D-%EB%B0%A9%EB%B2%95
# Kfold : K 개의 fold 를 만들어서 진행하는 교차검증(Cross Validation)
# 교차검증 : 데이터의 어느 정도를 훈련 데이터로 하고 또 어느정도를 테스트 데이터로 할지 결정하기는 쉽지 않다.
# 훈련데이터의 크기가 너무 작으면 모델링에 사용할 데이터가 적어 실제 도달 가능한 성능보다 낮은 성능의 모델이 만들어 질 것이다.
# 반대로 훈련데이터의 크기가 너무 커 테스트 데이터의 크기가 작아진다면 적은 수 의 데이터로만 테스트를 수행하게 되어 계산한 성능의 신뢰도가 낮아진다.
# 훈련데이터와 테스트 데이터의 분리방법도 문제가 된다. 우연히 테스트 데이터에서 성능이 잘 나오도록 데이터셋이 분리된다면 적절치 않은 평가가 될 것이다.
# 이러한 문제들을 개선하는 한 가지 방법이 교차검증이다.
# 교차검증은 데이터셋을 훈련데이터와 검증데이터(validation data) 로 나누어 모델링 및 평가하는 작업을 K회 반복하는 것으로, 이를 K겹 교차검증(K-fold Cross Valitation) 이라 한다.
# 보통 K 값은 10으로 지정한다.

# n_folds : 나눌 부분집합의 갯수(K) 를 지정, random_state : random 법칙 (set.Seed 와 비슷), Shuffle : 여기선 안썼지만 데이터셋 분할 전 Shuffle 을 할지 결정
kf = KFold(ntrain, n_folds=NFOLDS, random_state=SEED)

for i, (train_index, test_index) in enumerate(kf):
    print(i, test_index, test_index)


class SKlearnHelper(object):
    def __init__(self, clf, seed=0, params=None):  # clf : classifier
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        return self.clf.fit(x_train, y_train)  # fitting is equal to training

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self, x, y):
        return self.clf.fit(x, y)

    def feature_importances(self, x, y):
        print(self.clf.fit(x, y).feature_importances_)


import seaborn as sns


def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain))
    oof_test = np.zeros((ntest))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)

    # 배열.reshape(-1, 정수) 형식에서는 행(row) 의 개수가 가변적을 으로 정해진다
    # 배열의 길이가 12일 때 배열.reshape(-1, 4)   >> (3, 4) 가 만들어진다
    # 배열의 길이가 12일 때 배열.reshape(-1, 1)   >> (12, 1) 가 만들어진다.
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# Random Forest Parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
    'warm_start': True,
    # 'max_features' : 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators': 500,
    # 'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost Parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate': 0.75
}

# Gradient Boosting Parameters
gb_params = {
    'n_estimators': 500,
    # 'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier Parameters
svc_params = {
    'kernel': 'linear',
    'C': 0.025
}

rf = SKlearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SKlearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SKlearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SKlearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SKlearnHelper(clf=SVC, seed=SEED, params=svc_params)

# ravel : 다차원배열을 1차원 배열로 만들어주는 Numpy 함수
# 배열.drop() >> 특정 열을 삭제한 객체를 리턴, 디폴트인 axis=0 은 행(index) 을 삭제한다는 의미이며 axis=1  일땐 열을 삭제한다는 의미
y_train = train['Survived'].ravel()
train = train.drop(['Survived'], axis=1)
x_train = train.values
x_test = test.values

et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test)  # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf, x_train, y_train, x_test)  # Random Forest
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test)  # AdaBoost
gb_oof_train, gb_oof_test = get_oof(gb, x_train, y_train, x_test)  # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc, x_train, y_train, x_test)  # Support Vector

print("Training is complete")

rf_feature = rf.feature_importances(x_train, y_train)
et_feature = et.feature_importances(x_train, y_train)
ada_feature = ada.feature_importances(x_train, y_train)
gb_feature = gb.feature_importances(x_train, y_train)

# DataFrame : pandas 의 자료구조
# Pandas의 Series 가 1차원 형태의 자료구조라면 DataFrame 은 여러개의 칼럼(Column) 으로 구성된 2차원 형태의 자료구조


# 선언방법
# raw_data = {'col0': [1, 2, 3, 4],
#             'col1': [10, 20, 30, 40],
#             'col2': [100, 200, 300, 400]}

# 결과물
#     col0  col1  col2
# 0     1    10   100
# 1     2    20   200
# 2     3    30   300
# 3     4    40   400


# Ploty 패키지를 통해 feature importance 를 잘 구분할 수 있도록 DataFrame 으로 만든다
cols = train.columns.values

print('cols: ', cols)

# feature importances 를 DataFrame 으로 구성
feature_dataframe = pd.DataFrame({'features': cols,
                                  'Random Forest feature importances': rf_feature,
                                  'Extra Trees feature importances': et_feature,
                                  'AdaBoost feature importances': ada_feature,
                                  'Gradient Boost feature importances': gb_feature
                                  })

print(rf_feature)
print(et_feature)
print(ada_feature)
print(gb_feature)

print(feature_dataframe['Random Forest feature importances'].values)
print(feature_dataframe['Extra Trees feature importances'].values)
print(feature_dataframe['AdaBoost feature importances'].values)
print(feature_dataframe['Gradient Boost feature importances'].values)

print(feature_dataframe['features'])
# # Scatter plot
# trace = go.Scatter(
#     y=feature_dataframe['Random Forest feature importances'].values,
#     x=feature_dataframe['features'].values,
#     mode='markers',
#     marker=dict(
#         sizemode='diameter',
#         sizeref=1,
#         size=25,
#         color=feature_dataframe['Random Forest feature importances'].values,
#         colorscale='Portland',
#         showscale=True
#     ),
#     text=feature_dataframe['features'].values
# )
# data = [trace]
#
# layout= go.Layout(
#     autosize= True,
#     title= 'Random Forest Feature Importance',
#     hovermode= 'closest',
#
#     yaxis=dict(
#         title= 'Feature Importance',
#         ticklen= 5,
#         gridwidth= 2
#     ),
#     showlegend= False
# )
# fig = go.Figure(data=data, layout=layout)
# py.iplot(fig,filename='scatter2010')