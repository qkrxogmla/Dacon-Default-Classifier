import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_score
import joblib
import os

# 1. 전처리된 train 데이터 불러오기
train_df = pd.read_csv('data/train_preprocessed.csv')

# 2. 입력 데이터(X)와 타깃 변수(y) 분리
X = train_df.drop(columns=['UID', '채무 불이행 여부'])  
y = train_df['채무 불이행 여부']

# 3. 5-Fold 교차 검증 설정
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 4. LightGBM 모델 정의
model = lgb.LGBMClassifier(
    n_estimators=500,  # 500개의 부스팅 트리
    max_depth=8,  # 트리 깊이
    learning_rate=0.05,  # 학습률
    objective='binary',  # 이진 분류
    metric='auc',  # 평가 지표 (AUC 사용)
    random_state=42
)

# 5. 교차 검증 수행 및 결과 출력
scores = cross_val_score(model, X, y, cv=kf, scoring='roc_auc')  # AUC 점수 사용
print("5-Fold Cross Validation AUC Scores:", scores)
print("평균 AUC:", np.mean(scores))

# 6. 전체 데이터로 최종 모델 학습
print("📌 LightGBM 모델 학습 시작...")
model.fit(X, y)

# 7. 모델 저장 폴더 생성 (없으면 생성)
os.makedirs('models', exist_ok=True)

# 8. 학습된 모델 저장
joblib.dump(model, 'models/lightgbm_model.pkl')
print("✅ LightGBM 모델이 'models/lightgbm_model.pkl'에 저장되었습니다.")
