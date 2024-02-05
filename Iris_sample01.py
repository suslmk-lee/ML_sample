from sklearn import datasets  #머신러닝 알고리즘을 시험해 볼수 있는 표준 데이터셋을 제공한다.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 붓꽃 데이터셋을 로드한다. 
iris = datasets.load_iris()  #표준 데이터셋을 로딩한다. 
X = iris.data                #데이터 특성(features) 
y = iris.target              #레이블(labels)

# 데이터를 훈련세트와 테스트 세트로 분리한다. 
# train_test_split 함수는 데이터셋을 훈련셋과 테스트셋으로 분류해 준다.
# test_size : 전체 데이터셋중 0.3(30%)를 테스트 셋트로 사용하겠다는 의미이다.
# random_state : 실행할때마다 동일한 결과를 얻기 위한 난수시드이다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# 데이터 표준화
sc = StandardScaler()                 #특성들이 같은 스케일을 갖도록 조정한다.
sc.fit(X_train)                       # fit()메소드는 훈련 데이터의 평균과 표준편차를 계산해 준다.
X_train_std = sc.transform(X_train)   # transform()메소드는 이 평균과 표준편차를 사용하여 데이터를 표준화 한다. 
X_test_std = sc.transform(X_test)    

# k-최근접 이웃(KNN) 모델을 생성하고 훈련한다.
knn = KNeighborsClassifier(n_neighbors=5) #KNeighbors알고리즘을 사용한 분류기를 생성, n_neighbors=5 는 가장 가까운 5개의 이웃을 보고 분류를 결정한다.
knn.fit(X_train_std, y_train)             #fit()메소드는 모델을 훈련 데이터에 맞춘다.

# 테스트 데이터로 모델을 평가한다.
y_pred = knn.predict(X_test_std)                          #predict()메소드는 테스트 데이터에 대한 예측을 수행한다.
print("Accuracy: %.2f" % accuracy_score(y_test, y_pred))  #accuracy_score()함수는 모델의 정확도를 계산한다.