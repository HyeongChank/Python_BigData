import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 데이터 로드
titanic = pd.read_csv('./data/titanic.csv')

# 생존 여부를 0, 1로 변환
titanic['survived'] = titanic['survived'].replace({0: 'No', 1: 'Yes'})

# 사용할 변수 선택
X = titanic[['pclass', 'sex', 'age', 'fare']]
y = titanic['survived']

# 범주형 변수를 더미 변수로 변환
X = pd.get_dummies(X, columns=['pclass', 'sex'])

# 훈련 데이터와 테스트 데이터로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 로지스틱 회귀 모델 학습
model = LogisticRegression()
model.fit(X_train, y_train)

# 모델 예측 및 정확도 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))