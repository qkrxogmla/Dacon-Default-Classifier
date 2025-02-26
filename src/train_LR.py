import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

# 1. 전처리된 train 데이터 불러오기
train_df = pd.read_csv('data/train_preprocessed.csv')

# 2. 입력 데이터(X)와 타깃 변수(y) 분리
X = train_df.drop(columns=['UID', '채무 불이행 여부'])
y = train_df['채무 불이행 여부']

# 3. 🚀 데이터 스케일링 적용 (로지스틱 회귀에는 필수)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 5-Fold 교차 검증 설정
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 5. Logistic Regression 모델 정의 (불균형 데이터 대응)
model = LogisticRegression(
    solver='liblinear',
    class_weight='balanced',  # 🚀 불균형 데이터 자동 보정
    random_state=42
)

# 6. 교차 검증 수행 및 결과 출력
scores = cross_val_score(model, X_scaled, y, cv=kf, scoring='roc_auc')  # AUC 점수 사용
print("5-Fold Cross Validation AUC Scores:", scores)
print("평균 AUC:", np.mean(scores))

# 7. 전체 데이터로 최종 모델 학습
print("📌 Logistic Regression 모델 학습 시작...")
model.fit(X_scaled, y)

# 8. 모델 저장 폴더 생성 (없으면 생성)
os.makedirs('models', exist_ok=True)

# 9. 학습된 모델과 스케일러 저장
joblib.dump((model, scaler), 'models/logistic_model.pkl')
print("✅ Logistic Regression 모델이 'models/logistic_model.pkl'에 저장되었습니다.")
