import pandas as pd
import joblib

# 1. 전처리된 test 데이터 불러오기 (UID는 이미 제거된 상태)
test_df = pd.read_csv('data/test_preprocessed.csv')

# 2. 학습된 모델 불러오기
model = joblib.load('models/random_forest_model.pkl')

# 3. 예측 수행
predictions = model.predict(test_df)

# 4. 예측 결과를 DataFrame으로 정리 후 submissions 폴더에 저장
results = pd.DataFrame({'Prediction': predictions})
results.to_csv('submissions/test_predictions.csv', index=False)

print("Test predictions saved to submissions/test_predictions.csv")
