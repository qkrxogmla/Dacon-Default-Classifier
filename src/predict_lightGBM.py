import pandas as pd
import joblib
import os

# 1. 저장된 LightGBM 모델 불러오기
model = joblib.load('models/lightgbm_model.pkl')

# 2. 전처리된 테스트 데이터 불러오기
test_features = pd.read_csv('data/test_preprocessed.csv')

# 3. 원본 test 데이터에서 UID 불러오기
test_original = pd.read_csv('data/test.csv')
uids = test_original['UID']

# 4. UID 개수와 test_features 개수 확인
if len(uids) != len(test_features):
    print(f"⚠️ UID 개수({len(uids)})와 test 데이터 개수({len(test_features)})가 다릅니다.")
    exit()

# 5. 모델로 테스트 데이터 예측 (0 또는 1 반환)
pred_labels = model.predict(test_features)  # 🚀 0 또는 1 반환

# 6. 결과 저장 폴더 생성 (없으면 생성)
os.makedirs('submissions', exist_ok=True)

# 7. UID와 예측 결과를 결합하여 CSV 파일 저장
submission = pd.DataFrame({
    'UID': uids,
    '채무 불이행 확률': pred_labels  # 🚀 0 또는 1 값 저장
})

# 제출 형식에 맞게 컬럼명 수정 (필요 시 적용)
submission.rename(columns={'채무 불이행 확률': '채무 불이행 확률'}, inplace=True)

print("📌 CSV 저장을 시도합니다...")
submission.to_csv('submissions/test_predictions_lightgbm.csv', index=False)
print("✅ CSV 저장 완료: 'submissions/test_predictions_lightgbm.csv'")

# 8. 저장된 파일 확인
if os.path.exists("submissions/test_predictions_lightgbm.csv"):
    print("✅ 파일이 정상적으로 저장되었습니다!")
else:
    print("⚠️ 파일이 저장되지 않았습니다. 경로를 확인하세요.")
