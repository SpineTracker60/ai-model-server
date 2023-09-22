import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import re
# 모델 라이브러리 import
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier  # 분류


# SQL 파일 경로
sql_file = 'real mock data.sql'

def sql_to_csv(sql_script):
    query_pattern = r"INSERT INTO `` \(`id`,.*? VALUES \(.*?\);"

    # SQL 쿼리 추출
    queries = re.findall(query_pattern, sql_script, re.DOTALL)

    # 추출한 쿼리를 DataFrame으로 변환
    data = []
    for query in queries:
        values_start = query.index("VALUES") + 7
        values = query[values_start:].strip("();")
        values = values.split(',')

        # 따옴표를 제거하고 NULL 값을 원하는 형식으로 처리
        values = [value.strip("'").strip() if value != "NULL" else None for value in values]
        data.append(values)

    # DataFrame 생성
    df = pd.DataFrame(data, columns=["id", "gender", "birth_date", "job", "TEXTNECK", "ASYMMETRY", "STOOPED",
                                     "SLEEPINESS", "product_chair", "product_monitor_arm", "product_wrist_guard",
                                     "product_mouse", "product_desk", "product_holder", "product_neck_pillow",
                                     "product_mouse_pad", "product_keyboard", "product_cushion"])

    return df

# SQL 파일 읽기
with open(sql_file, 'r', encoding='utf-8') as sql_file:
    sql_script = sql_file.read()

# SQL 데이터를 DataFrame으로 변환
df = sql_to_csv(sql_script)

# 'product_id' 열을 추가하고, 각 행에서의 product 값을 저장
df['product_id'] = df.apply(lambda row: row['product_chair'] or row['product_monitor_arm'] or
                                     row['product_wrist_guard'] or row['product_mouse'] or
                                     row['product_desk'] or row['product_holder'] or
                                     row['product_neck_pillow'] or row['product_mouse_pad'] or
                                     row['product_keyboard'] or row['product_cushion'], axis=1)
# CSV 파일로 저장
csv_file = 'output.csv'
df.to_csv(csv_file, index=False)

print(f'Data converted from SQL file to CSV and saved as {csv_file}')

#저장한 csv 파일 로드
data = pd.read_csv('output.csv', encoding='utf-8')

def convert_age(birth_date):
    birth_year = int(birth_date.split('-')[0])
    current_year = 2023
    birth_year = current_year - birth_year
    if birth_year < 20:
        return "10's"
    elif birth_year < 30:
        return "20's"
    elif birth_year < 40:
        return "30's"
    elif birth_year < 50:
        return "40's"
    else:
        return "over50's"

data['birth_date'] = data['birth_date'].apply(convert_age)
data['product_id'] = data['product_id'].apply(str)

# 필요없는 열 'id' 제거
data = data.drop(columns=['id'])

# 원핫 인코딩
data = pd.get_dummies(data, columns=['gender','birth_date','job'])
product_name = ['product_chair','product_monitor_arm','product_wrist_guard','product_mouse','product_desk','product_holder','product_neck_pillow','product_mouse_pad','product_keyboard','product_cushion','product_id']
for P_name in product_name:
    Temp = data.dropna(subset=[P_name])

    X = Temp.drop(columns=product_name)
    y = Temp[P_name].astype(str)

    # 학습 데이터와 테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=777)

    feature_names = X.columns
    pipelines = {
      'lr': make_pipeline(StandardScaler(), LogisticRegression()),
      'rf': make_pipeline(StandardScaler(), RandomForestClassifier()),
      'svd': make_pipeline(StandardScaler(), SVC(probability=True)),
      'knc': make_pipeline(StandardScaler(), KNeighborsClassifier()),
      'gbm': make_pipeline(StandardScaler(), GradientBoostingClassifier()),
    }

    # 모델 학습 및 저장
    fit_models = {}
    for algo, model in pipelines.items():
        model.fit(X_train, y_train)
        fit_models[algo] = model

    # 모델 평가
    max_accuracy = 0
    best_model = None

    for algo, model in fit_models.items():
        yhat = model.predict(X_test)
        accuracy = accuracy_score(y_test, yhat)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_model = algo
        print(algo, accuracy)

    print(f'{P_name}모델중 가장 높은 성능을 보인 모델은 {best_model}입니다.')

    # 가장 높은 성능을 보인 모델을 파일로 저장
    with open(f'models/{P_name}_model.pkl', 'wb') as f:
        pickle.dump(fit_models[best_model], f)
