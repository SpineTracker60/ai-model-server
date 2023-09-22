from pathlib import Path
import json
from konlpy.tag import Okt

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory

class ChatBot():

    # 생성자
    def __init__(self) -> None:

        # 프롬프트 
        self.prompt = self.get_prompt()
        # 모델
        self.model = self.get_model()
        # 형태소 분석기
        self.okt = Okt()
    
    # 모델 로드
    def get_model(self) -> ChatOpenAI:
        
        # resource 폴더 안의 API 키 (.gitignore 설정)
        # root = Path.cwd()
        # key_dir = str(root) + '/SpineTracker60/resource'
        # print("key_dir :",key_dir)

        with open('resource/secret.json', 'r') as json_file:
            secret_json = json.load(json_file)
            
        openai_api_key = secret_json["OPENAI_API_KEY"]
        
        # 1) GPT 3.5 (*권장)
        defalut_model = "gpt-3.5-turbo"
        # 2) GPT 3.5 Fine-tunning (custom데이터 10개 학습)
        fine_tunning_model = secret_json["FINE_TUNNING_MODEL"]
        
        chat_model = ChatOpenAI(model=defalut_model, openai_api_key=openai_api_key)
        conversation = ConversationChain(
            
            # 프롬프트 템플릿 적용
            prompt=self.prompt,
            # 모델 적용
            llm=chat_model,
            verbose=False,
            # 대화 내용 기억 (k번까지)
            memory=ConversationBufferWindowMemory(memory_key='history', ai_prefix="AI Assistant", k=5)
        )
        return conversation
                #     만약 사용자의 질문이 "목배게 추천해줘" 와 같은 의미라면, "목배게 추천해드리겠습니다!"라고 말해줘. 이와 유사하게 "허리에 좋은 의자좀 알려줘"라고 질문한다면
                # "의자 추천해드리겠습니다."라고 말해줘. 또 유사하게 "손목에 좋은 마우스 추천해봐"라고 한다면 "마우스 추천해드리겠습니다!"라고 말해줘.

    # 프롬프트 템플릿
    def get_prompt(self) -> ChatPromptTemplate:
        template = """
                너는 '척추적 60분' 이라는 자세 교정 서비스의 챗봇이야.

                너가 최우선으로 할일은 사용자의 질문이 상품을 추천해 달라는 의미인지 아닌지 분류하는 것이야.

                사용자의 질문이 "목배게 추천해줘" 이거나 "허리에 좋은 의자좀 알려줘" 혹은 "손목에 좋은 마우스 추천해봐"와 같이 상품을 추천해달라는 질문이 입력될 수 있어.
                이때, 사용자가 원하는 물품+"추천해드리겠습니다!"라고 한문장으로만 답변해줘.

                만약 사용자의 질문이 상품을 추천해달라는 의미가 아니라면 너가 할일은 다음과 같아.
                '척추적 60분' 서비스는 컴퓨터 앞에서 공부하거나 작업할 때 사용하는 서비스이고 실시간으로 노트북 웹캠을 통해 앉은 자세를 감지하면서 자세 비대칭, 거북목, 졸음 현상을 
                발견했을 때 유저에게 알림을 보내주는 서비스야. 사용자들이 너에게 '척추적 60분' 서비스의 사용법이나 스트레칭 방법에 대한 질문을 할 수 있고 너는 이에 대해 답변해주어야해. 

                대표적으로 사용자들이 할 수 있는 질문은 다음과 같아.
                첫번째로 "요정만들기는 무엇인가요?"야. 이때는 서비스 사용하므로써 쌓이는 척추 모형을 토대로 척추 요정을 만드는 버튼이라고 하면돼.
                두번째로 "아까 찍은 사진은 어디에 활용되나요?"야. 이때는 거북목,자세 비대칭,졸음 감지를 위해 수집한 데이터이고 이외에 목적으로는 활용하지 않는다라고 하면돼.
                세번째로 "척추(모형)은 몇분마다 생성되나요?"야. 이때는 척추(모형)이 30분마다 1개씩 생성되며 마이페이지를 통해 일간,주간,월간 통계를 확인할 수 있다고 하면돼.
                네번째로 "화장실 가려는데 정지할 수 있나요?"야. 이때는 서비스 이용 중에 자리를 잠시 이탈해야하는 볼일이 있다면 화면 좌측 상단의 일시정지 버튼을 눌러 자세 감지를 잠깐동안 중지할 수 있다고 하면돼.

                질문에 대해 답변할 때는 이모티콘을 많이 활용해서 귀엽고 깜찍한 말투로 말해주면 좋겠어. 가끔은 농담 섞인 답변으로 해도 좋아.
                너의 이름은 한국의 대표적인 성씨인 '김'씨와 스켈레톤의 '레톤'을 결합한 '김레톤'이야. 

                모든 답변은 20자 이내로 대답해줘.

                Current conversation:
                {history}
                Human: {input}
                AI Assistant:"""

        prompt = PromptTemplate(input_variables=["history","input"], template=template)
        return prompt

    # 답변 추론
    def get_answer(self, text) -> str:
        return self.model.predict(input=text)

# 테스트 Main 함수        
if __name__ == '__main__':
    chatbot = ChatBot()

    while True:

        text = input("Human >> ")

        # 테스트 중단 시 'quit' 입력
        if 'quit' in text:
            break
        
        result = chatbot.get_answer(text)

        # 상품 추천일 때
        if '추천' in result:
            
            # 형태소 분석기를 통한 상품 종류 key 추출
            # ex) 허리 편한 의자 추천좀! => '의자' 추출
            # 의자,마우스,모니터 거치대 등 상품 분류 리스트 필요!
            result += (" 상품 분류 : " + chatbot.okt.nouns(result)[-2])
        print("Chabot >> " + result)