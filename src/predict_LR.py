import pandas as pd
import joblib
import os

# 1. 저장된 Logistic Regression 모델 & 스케일러 불러오기
model, scaler = joblib.load('models/logistic_model.pkl')

# 2. 전처리된 테스트 데이터 불러오기
test_features = pd.read_csv('data/test_preprocessed.csv')

# 3. 원본 test 데이터에서 UID 불러오기
test_original = pd.read_csv('data/test.csv')
uids = test_original['UID']

# 4. 🚀 테스트 데이터도 같은 방식으로 스케일링
test_features_scaled = scaler.transform(test_features)

# 5. 모델로 테스트 데이터 예측 (0 또는 1 반환)
pred_labels = model.predict(test_features_scaled)  # 🚀 0 또는 1 값 저장

# 6. 결과 저장 폴더 생성 (없으면 생성)
os.makedirs('submissions', exist_ok=True)

# 7. UID와 예측 결과를 결합하여 CSV 파일 저장
submission = pd.DataFrame({
    'UID': uids,
    '채무 불이행 확률': pred_labels
})

print("📌 CSV 저장을 시도합니다...")
submission.to_csv('submissions/test_predictions_logistic.csv', index=False)
print("✅ CSV 저장 완료: 'submissions/test_predictions_logistic.csv'")

# 8. 저장된 파일 확인
if os.path.exists("submissions/test_predictions_logistic.csv"):
    print("✅ 파일이 정상적으로 저장되었습니다!")
else:
    print("⚠️ 파일이 저장되지 않았습니다. 경로를 확인하세요.")
