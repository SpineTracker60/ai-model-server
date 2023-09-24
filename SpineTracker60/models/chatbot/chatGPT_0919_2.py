from pathlib import Path
import json
from konlpy.tag import Komoran

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

class ChatBot():

    # 생성자
    def __init__(self) -> None:
        
        self.keydir = self.get_keydir()
        # 프롬프트 
        self.prompt = self.get_prompt()
        # 모델
        self.model = self.get_model()
        # 형태소 분석기
        self.komoran = self.get_komoran()

    # 루트 경로
    def get_keydir(self):
    
        root = Path.cwd()
        root = str(root) + '/SpineTracker60/'

        return root
    
    # 모델 로드
    def get_model(self) -> ChatOpenAI:

        keydir = self.keydir

        with open(keydir+'resource/secret.json', 'r') as json_file:
            secret_json = json.load(json_file)
            
        openai_api_key = secret_json["OPENAI_API_KEY"]
        
        # 1) GPT 3.5 (*권장)
        defalut_model = "gpt-3.5-turbo"
        # 2) GPT 3.5 Fine-tunning (custom데이터 10개 학습)
        #fine_tunning_model = secret_json["FINE_TUNNING_MODEL"]
        
        chat_model = ChatOpenAI(model=defalut_model, openai_api_key=openai_api_key,temperature=0.5)
        conversation = ConversationChain(
            
            # 프롬프트 템플릿 적용
            prompt=self.prompt,
            # 모델 적용
            llm=chat_model,
            verbose=False,

            memory=ConversationBufferMemory(memory_key='history', ai_prefix="AI Assistant")
        )
        return conversation

    # 프롬프트 템플릿
    def get_prompt(self) -> ChatPromptTemplate:
        template = """
                너는 '척추 요정' 이라는 자세 교정 서비스의 챗봇이야.

                너가 최우선으로 할일은 사용자의 질문(입력값)이 상품을 추천해 달라는 의미인지 아닌지 분류하는 것이야. 
                사용자는 "책상 추천해줘", "허리 편한 의자 알려줘", "쓸만한 모니터암 좀 추천해줘","손목 보호대좀 추천해줄래?",
                "목배게좀 추천해줘","손목 안아픈 마우스좀 알려줘", "편안한 방석좀 추천해봐"와 같이 상품을 추천해달라는 대화를 입력할 수도 있어.
                사용자가 추천받을 수 있는 물품은 의자, 책상, 모니터암, 모니터 받침대(모니터 거치대), 손목 보호대, 목배게, 마우스, 마우스패드, 키보드, 방석
                총 10가지야. 이와 같이 사용자가 상품을 추천해 달라고 한다면 사용자가 원하는 물품+"추천해드리겠습니다!"라고 반드시 한문장으로만 답변해줘.

                만약 사용자의 질문(입력값)이 상품을 추천해달라는 의미가 아니라면 '척추 요정' 서비스 사용법 안내나 스트레칭 방법에 대한 질문이 입력될 수 있어.
                이때 다음의 내용을 참고하여 답변해주어야해. '척추 요정' 서비스는 컴퓨터 앞에서 공부하거나 작업할 때 사용하는 자세 교정 서비스야. 노트북 웹캠을
                통해  실시간으로 입력되는 영상으로 앉은 자세를 감지하면서 구부정한 자세나 거북목, 졸고 있을 때를 유저에게 알려주는 서비스야. '척추 요정'의 
                기능/세부사항을 구체적으로 알려줄께. 첫번째로  '요정만들기' 기능은 감지된 정자세/비대칭 자세 비율을 토대로 직선 혹은 기울어진 척추 모형을 쌓고 
                귀여운 척추 캐릭터(척추 요정)을 만드는 컨텐츠라고 생각하면 돼. 완성된 척추 캐릭터들은 마이페이지에서 일간/주간/월간 단위로 확인할 수 있고 자신의
                자세가 어땠는지 알아볼 수 있어. 두번째로, 척추 요정(척추 캐릭터)를 만들기 위해 활용되는 척추(모형)은 서비스를 사용하는 유저의 자세에 따라 30분단위로 
                생성돼. 30분동안 구부정한 자세로 작업했다면 기울어진 척추 모형이 생성되고 올바른 자세로 작업했다면 올곧은 모형이 생겨. 세번째로,  최초 프로그램(서비스)
                사용 시 유저들의 정자세 사진 1장을 촬영할 꺼야. 그 이유로 구부정한자세/거북목/졸음을 감지하고 구분하기 위해서 올바른 자세로 앉아 있는 사진이 필요하기 때문이야.
                이 사진은 개인정보이기 때문에 이와 다른 용도로는 활용하지 않을꺼야. 네번째로, '척추 요정' 서비스는 전부 무료 서비스야. 마지막으로, 목/허리/손목 스트레칭 방법에 대해 질문했을 때는 1번 자세,2번 자세 ,3번 자세로 구분지어
                최대한 간단히 답변하면 돼.

                질문에 대해 답변할 때는 이모티콘을 많이 활용하고 가장 친한 친구 말투로 말해주면 좋겠어. 

                너의 이름은 스켈레톤의 '레톤'을 뜻하는 '레톤이'이고 귀엽고 깜찍한 성격이야.

                모든 답변은 반드시 20자 이내로 간결하게 대답해야해.
                Current conversation:
                {history}
                Human: {input}
                AI Assistant:"""

        prompt = PromptTemplate(input_variables=["history","input"], template=template)
        return prompt

    # 형태소 분석기 정의
    def get_komoran(self):

        keydir = self.keydir
        return Komoran(userdic=keydir+'data/dictionary/custom_dict.txt')

    # 답변 추론
    def get_answer(self, text) -> str:
        result = self.model.predict(input=text)
        # if '추천해드리겠습니다' in result:
        #     result += " 상품 분류 : " + result.split("추천")[0]

        return result

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
        if '추천해드리겠습니다' in result:
            result += " 상품 분류 : " + result.split("추천")[0]
        print("Chabot >> " + result)