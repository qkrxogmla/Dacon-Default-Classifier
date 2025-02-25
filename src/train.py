import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib  # 모델 저장을 위해 joblib을 임포트

# 미리 분할된 데이터 파일 불러오기
X_train = pd.read_csv('data/0.8info.csv')
y_train = pd.read_csv('data/0.8ans.csv')
X_val = pd.read_csv('data/0.2info.csv')
y_val = pd.read_csv('data/0.2ans.csv')

# 만약 y_train, y_val 파일이 단일 컬럼이라면, np.ravel()이나 .squeeze()로 1차원 배열로 만들어주세요.
y_train = y_train.squeeze()
y_val = y_val.squeeze()

# 모델 학습
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))

# 학습된 모델을 models 폴더에 저장 (폴더가 존재해야 합니다)
joblib.dump(model, 'models/random_forest_model.pkl')
