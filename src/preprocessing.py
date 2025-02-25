import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# 데이터 불러오기
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# 필요한 전처리 작업 수행 (예: 컬럼 변환, 인코딩 등)
categorical_col = ['주거 형태', '현재 직장 근속 연수', '대출 목적', '대출 상환 기간']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder.fit(train_df[categorical_col])

train_encoded = encoder.transform(train_df[categorical_col])
test_encoded = encoder.transform(test_df[categorical_col])

train_encoded_df = pd.DataFrame(train_encoded, columns=encoder.get_feature_names_out(categorical_col))
test_encoded_df = pd.DataFrame(test_encoded, columns=encoder.get_feature_names_out(categorical_col))

# 전처리된 train 데이터: 원본 데이터에서 범주형 컬럼 제거 후 인코딩 결과와 결합
train_preprocessed = pd.concat([train_df.drop(columns=categorical_col).reset_index(drop=True), train_encoded_df], axis=1)
# 전처리된 test 데이터: 원본 데이터에서 범주형 컬럼 제거 후 인코딩 결과와 결합
test_preprocessed = pd.concat([test_df.drop(columns=categorical_col).reset_index(drop=True), test_encoded_df], axis=1)

# train 데이터 전체를 저장 (여기서 UID와 타깃은 그대로 유지)
train_preprocessed.to_csv('data/train_preprocessed.csv', index=False)

# test 데이터에서 UID 컬럼 제거 후 저장
test_preprocessed.drop(columns=['UID'], inplace=True)
test_preprocessed.to_csv('data/test_preprocessed.csv', index=False)

print("전처리 완료 및 CSV 파일 저장됨.")
