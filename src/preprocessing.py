import pandas as pd
from sklearn.model_selection import train_test_split
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

train_df = pd.concat([train_df.drop(columns=categorical_col).reset_index(drop=True), train_encoded_df], axis=1)
test_df = pd.concat([test_df.drop(columns=categorical_col).reset_index(drop=True), test_encoded_df], axis=1)

# 타깃 컬럼 "채무 불이행 여부"를 기준으로 데이터 분리 (여기서 컬럼명이 정확한지 확인)
X_train, X_val, y_train, y_val = train_test_split(
    train_df.drop(columns=['UID', '채무 불이행 여부']), 
    train_df['채무 불이행 여부'], 
    test_size=0.2, 
    random_state=42
)

X_train.to_csv('data/0.8info.csv', index=False)
X_val.to_csv('data/0.2info.csv', index=False)
y_train.to_csv('data/0.8ans.csv', index=False)
y_val.to_csv('data/0.2ans.csv', index=False)

# UID 컬럼 제거한 후 저장
test_df_no_uid = test_df.drop(columns=['UID'])
test_df_no_uid.to_csv('data/test_preprocessed.csv', index=False)
