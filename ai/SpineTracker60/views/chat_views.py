from flask import Blueprint,request,jsonify
import pandas as pd
import pickle
import sys
import os
import requests

# 현재 스크립트 파일의 경로를 얻어옵니다.
current_folder = os.path.dirname(os.path.abspath(__file__))
# 프로젝트 루트 디렉토리를 계산합니다.
project_root = os.path.dirname(current_folder)
# 프로젝트 루트 디렉토리를 sys.path에 추가합니다.
sys.path.append(project_root)

from models.chatbot.chatGPT_0919_2 import ChatBot

# 챗봇 bp
bp = Blueprint('chat',__name__,url_prefix='/chat')

chatbot = ChatBot()

# 사용자 질문 종류

# 1) 시스템 안내 -> (chatGPT)
# 2) 스트레칭 질문 -> (chatGPT)
# 3) 상품 추천 -> (chatGPT) *chatGPT 의도 분류 + (ML) 추천 상품 추론
# 4) 기타 -> (chatGPT)

#Recommendaton system

def convert_age(birth_date):
    birth_year = int(birth_date.split('-')[0])
    current_year = 2023
    birth_year = current_year - birth_year
    if birth_year < 20:
        return "birth_date_10\'s"
    elif birth_year < 30:
        return "birth_date_20\'s"
    elif birth_year < 40:
        return "birth_date_30\'s"
    elif birth_year < 50:
        return "birth_date_40\'s"
    else:
        return "birth_date_over50\'s"

gender = {"FEMALE" : "gender_Female", "MALE" : "gender_Male"}
job = {'creator' : 'job_Creator', 'etc' : 'job_ETC', 'marketer' : 'job_Marketer', 'office worker' : 'job_Office worker',
       'designer' : 'job_designer', 'developer' : 'job_developer',
       'planner' : 'job_planner', 'student' : 'job_student'}
product_names = {'의자' : 'chair' , '쿠션' : 'cushion' , '책상' :'desk', '홀더': 'holder', '키보드' :'keyboard',
                 '모니터 암': 'monitor_arm', '마우스' : 'mouse', '마우스 패드' : 'mouse_pad', '목 베게' : 'neck_pillow',
                 '손목 보호대' : 'wrist_guard_model'}

def Rec(product, input_json):
    if product in product_names: #만약에 추천해달라한 제품이 리스트에 있다면
        product = product_names[product]
    else:                        #없다면 모든것 중에 추천
        product = 'id'
    # 저장된 모델을 불러오기
    with open(f'SpineTracker60/recommendation/models/product_{product}_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    user_input = {
        convert_age(input_json['birth_date']) : 1,
        gender[input_json['gender']] : 1,
        job[input_json['job']]: 1,
        'TEXTNECK': input_json['TEXTNECK'],
        'ASYMMETRY': input_json['ASYMMETRY'],
        'STOOPED': input_json['STOOPED'],
        'SLEEPINESS': input_json['SLEEPINESS'],
    }
    # 원래의 제품명
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
    print(f"회원님의 정보로 저희가 추천 드릴 수 있는 상위 {num_recommendations}개의 {product}제품은:",
          '번,' ''.join(map(str, [class_mapping[label] for label in top_product_indices])), '번 입니다.')
    return [class_mapping[label] for label in top_product_indices]
# 답변 
@bp.route('/answer',methods=['POST'])
def gen_ans():
    
    try:
        input_json = request.json
        question = input_json.pop('question')
        answer = chatbot.get_answer(question)
        answer_json = dict()
        answer_json['question'] = question
        answer_json['answer'] = answer
        if '추천' in answer:
            #print(chatbot.okt.nouns(answer)[-2])
            #answer += " 상품 분류 : " + chatbot.okt.nouns(answer)[-2]
            product = answer.split("추천")[0].strip()
            print("답변 : ",answer)
            answer_json['product'] = product
            answer_json['product_num'] = Rec(product,input_json)
        return jsonify(answer_json)

    except Exception as e:
        print("예외 발생!!! ",str(e))
        answer = str(e)
        return jsonify(answer)



# 답변 테스트
@bp.route('/answer_test',methods=['POST'])
def gen_ans_test():
    
    try:
        input_form = request.form
        user_input = input_form['question']
        
        print("질문 : ",user_input)
        answer = chatbot.get_answer(user_input)

        if '추천' in answer:
            answer += (" 상품 분류 : " + chatbot.okt.nouns(answer)[-2])

        print("답변 : ",answer)

    except Exception as e:
        print("예외 발생!!! ",str(e))
        answer = str(e)

    return answer


#example
{
     "id" : 1,
     "age": 27,
     "gender" : 1,
     "job" : 7,
     "TEXTNECK" : 0.2,
     "ASYMMETRY" : 0.5,
     "STOOPED" : 0.7,
     "SLEEPINESS" : 0.5,
     "question" : "의자 추천해줄래?"
 }