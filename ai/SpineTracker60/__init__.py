from flask import Flask
import json
import os
from dotenv import load_dotenv

# # 테스트용 json 파일 생성
# data = {
#     "id" : 1,
#     "age": 27,
#     "gender" : 1,
#     "job" : 7,
#     "turtle_neck" : 0.2,
#     "sleepiness" : 0.5,
#     "tooped_position" : 0.7,
#     "question" : '이거 무슨 프로그램이야'
# }

# # JSON 파일로 데이터 저장
# with open("tmp_user.json", "w", encoding="utf-8") as json_file:
#     json.dump(data, json_file, ensure_ascii=False,indent=7)

def create_app():
    app = Flask(__name__)

    from .views import main_views,chat_views,rs_views,board_views
    app.register_blueprint(main_views.bp)
    app.register_blueprint(chat_views.bp)
    app.register_blueprint(rs_views.bp)
    app.register_blueprint(board_views.bp)
    
    return app

# flask run --host=0.0.0.0 --port=5001

# ngrok 로그 확인
# http://127.0.0.1:4040