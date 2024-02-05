import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# CSV 파일을 읽어 온다. 이 예제에서는 'data.csv'를 준비한다.
# 'data.csv' 파일은 같은 디렉토리에 있어야 한다.
# 파일의 구조는 첫번째 행이 열 이름이고, 마지막열이 레이블이라고 가정한다.
df = pd.read_csv('data.csv')

# 'exercise'열을 수치형으로 변환한다. 'Yes'는 1, 'No'는 0으로 매핑한다.
df['exercise'] = df['exercise'].map({'Yes': 1, 'No': 0})

# 특성(X)과 레이블(y)를 분리한다.
X = df.iloc[:, :-1].values  # 마지막열을 제외한 모든 열
y = df.iloc[:, -1].values   # 마지막 행

# 데이터를 훈련세트와 테스트 세트로 분리한다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# 데이터 표준화
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# KNN 모델 생성 및 훈련
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_std, y_train)

# 테스트 데이터로 모델 평가
y_prod = knn.predict(X_test_std)
print("Accuracy: %.2f" % accuracy_score(y_test, y_prod))