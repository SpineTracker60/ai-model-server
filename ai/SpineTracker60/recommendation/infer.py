import pandas as pd
import pickle

#추론
# 추론할 정보
user_input = {
    'product' : 'chair',
    'gender_Female': 1,
    'birth_date_20\'s': 1,
    'job_Creator': 1,
    'TEXTNECK': 0.5,
    'ASYMMETRY': 0.3,
    'STOOPED': 0.7,
    'SLEEPINESS': 0.4
}

# 저장된 모델을 불러오기
product = user_input.pop("product")
with open(f'models/product_{product}_model.pkl', 'rb') as f:
  loaded_model = pickle.load(f)

#원래의 제품명
class_names = loaded_model.classes_
class_mapping = {i: str(int(float(class_names[i]))) for i in range(len(class_names))}

# 원핫인코딩을 했기 때문에 선택하지 않은 정보에 대해선 0으로 처리
user_df = pd.DataFrame([user_input])
empty_df = pd.DataFrame(columns=['TEXTNECK', 'ASYMMETRY', 'STOOPED', 'SLEEPINESS', 'gender_Female',
       'gender_Male', 'birth_date_10\'s', 'birth_date_20\'s', 'birth_date_30\'s',
       'birth_date_40\'s', 'birth_date_over50\'s', 'job_Creator', 'job_ETC',
       'job_Marketer', 'job_Office worker', 'job_designer', 'job_developer',
       'job_planner', 'job_student'])
final_user_df = pd.concat([empty_df, user_df], ignore_index=True, sort=False).fillna(0)

# 제품 추천 확률 계산
predicted_probabilities = loaded_model.predict_proba(final_user_df)[0]
num_recommendations = 3  # 상위 3개 제품 추천
top_product_indices = predicted_probabilities.argsort()[::-1][:num_recommendations]
# 상위 N개 제품 추천
print(f"회원님의 정보로 저희가 추천 드릴 수 있는 상위 {num_recommendations}개의 {product}제품은:", '번,' ''.join(map(str,[class_mapping[label] for label in top_product_indices])),'번 입니다.')