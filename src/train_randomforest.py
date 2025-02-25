import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. 전처리된 train 데이터 불러오기 (UID와 타깃 포함)
train_df = pd.read_csv('data/train_preprocessed.csv')
# train_preprocessed.csv에는 전처리 과정이 완료된 데이터가 저장되어 있습니다.
# (예: preprocessing.py에서 UID와 '채무 불이행 여부'를 포함하여 저장)

# 2. 학습에 사용할 입력 데이터(X)와 타깃(y) 구성 (UID는 제거)
X = train_df.drop(columns=['UID', '채무 불이행 여부'])
y = train_df['채무 불이행 여부']

# 3. 5-Fold 교차 검증 수행
kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = RandomForestClassifier(random_state=42)

# cross_val_score()를 통해 각 fold의 정확도 평가
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
print("5-Fold Cross Validation Scores:", scores)
print("평균 정확도:", np.mean(scores))

# 4. 최종 모델 학습 (전체 훈련 데이터를 사용)
model.fit(X, y)

# 5. 학습된 최종 모델을 models 폴더에 저장 (폴더가 미리 존재해야 함)
joblib.dump(model, 'models/random_forest_model.pkl')