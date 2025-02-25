import pandas as pd
import joblib

# 6. 전처리된 test 데이터 불러오기 (UID는 제거되지 않은 원본 테스트 파일에서 추출)
# test_preprocessed.csv는 UID가 제거된 상태로 저장되어 있으므로, 예측 입력으로 사용합니다.
test_features = pd.read_csv('data/test_preprocessed.csv')

model = joblib.load('models/random_forest_model.pkl')
# 만약 제출 파일에 UID가 필요하다면, 원본 test.csv에서 UID 컬럼을 가져옵니다.
test_original = pd.read_csv('data/test.csv')
uids = test_original['UID']

# 7. 최종 모델로 테스트 데이터 예측 (클래스 1, 즉 채무 불이행 확률 예측)
pred_probs = model.predict(test_features)

# 8. UID와 예측 결과를 결합하여 submissions 폴더에 CSV 파일로 저장
submission = pd.DataFrame({
    'UID': uids,
    '채무 불이행 확률': pred_probs
})
print("📌 CSV 저장을 시도합니다...")
submission.to_csv('submissions/test_predictions_randomforest.csv', index=False)
print("✅ CSV 저장 완료!")
