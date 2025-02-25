import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
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

# 4. CatBoost 모델 정의 (자동 범주형 변수 인식)
model = CatBoostClassifier(iterations=500,  # 500번 반복 학습
                           depth=8,  # 트리 깊이
                           learning_rate=0.05,  # 학습률
                           loss_function='Logloss',
                           eval_metric='AUC',
                           verbose=100,  # 100번째 반복마다 로그 출력
                           random_seed=42)

# 5. 교차 검증 수행 및 결과 출력
scores = cross_val_score(model, X, y, cv=kf, scoring='roc_auc')  # AUC 점수 사용
print("5-Fold Cross Validation AUC Scores:", scores)
print("평균 AUC:", np.mean(scores))

# 6. 전체 데이터로 최종 모델 학습
print("📌 CatBoost 모델 학습 시작...")
model.fit(X, y, verbose=100)

# 7. 모델 저장 폴더 생성 (없으면 생성)
os.makedirs('models', exist_ok=True)

# 8. 학습된 모델 저장
model.save_model('models/catboost_model.cbm')
print("✅ CatBoost 모델이 'models/catboost_model.cbm'에 저장되었습니다.")
