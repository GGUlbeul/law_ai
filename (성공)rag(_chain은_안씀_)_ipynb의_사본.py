

### 라이브러리 다운로드
"""

#코랩에서 한다면 이걸로
!pip install langchain_openai
!pip install langchain
!pip install langchain_community
!pip install datasets
!pip install sentence_transformers
!pip install faiss-cpu
!pip install -U transformers
!pip install huggingface_hub
!pip install pypdf
!pip install beautifulsoup4
!pip install pandas
!pip install torch

from langchain_community.embeddings import HuggingFaceEmbeddings
import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from datasets import load_dataset

"""#faiss불러오기(법률)



"""

from huggingface_hub import login
import pandas as pd
login(os.get_env("hug_API_KEY"))

from huggingface_hub import hf_hub_download
import os

path_1 = '/content/test_1'
#교수님께서 경로 설정을 해주셔야 합니다!


for filenames in ['index.pkl','index.faiss']:
  test_1 = (hf_hub_download(repo_id='sungjinny/all_law_embedding', filename= filenames, cache_dir=path_1))


test_1 = os.path.dirname(test_1)

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
embeddings_model = HuggingFaceBgeEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")
vectorstors_1 = FAISS.load_local(test_1, embeddings_model, allow_dangerous_deserialization = True)


retriever_1 = vectorstors_1.as_retriever(search_type="similarity", search_kwargs = { 'k' :3})

"""#faiss불러오기 (판례)

"""

from huggingface_hub import login
import pandas as pd


from huggingface_hub import hf_hub_download
import os

path_2 = '/content/test_2'
#교수님께서 경로 설정을 해주셔야 합니다!


for filenames in ['index.pkl','index.faiss']:
  test_2 = (hf_hub_download(repo_id='sungjinny/precedent_embedding', filename= filenames, cache_dir=path_2))


test_2 = os.path.dirname(test_2)

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
embeddings_model = HuggingFaceBgeEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")
vectorstors_precedent = FAISS.load_local(test_2, embeddings_model, allow_dangerous_deserialization = True)


retriever_precedent = vectorstors_precedent.as_retriever(search_type="similarity", search_kwargs = { 'k' :3})

"""#gpt합치기 (법률 서비스, 판례 서비스)찾기

"""
from dotenv import load_dotenv
import os

load_dotenv()  # .env 파일을 로드

os.environ['OPENAI_API_KEY'] = os.getenv("API_KEY")


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# Commented out IPython magic to ensure Python compatibility.
# # open ai 업데이트되면서 이거 해야함 나중에는 뺄 것
# %%capture
# !pip install httpx==0.27.2 --force-reinstall --quiet

import time
from langchain_core.prompts import PromptTemplate

# 법률 관련 기능
def explain_law(search_result):
    """법 내용을 쉽게 요약하여 설명"""
    law_name = search_result.metadata
    law_content = search_result.page_content
    prompt = PromptTemplate.from_template("당신은 전문변호사 입니다. {law_name}과 {law_content}를 일반인이 보고 쉽게 이해할 수 있도록 2줄로 요약해주세요")

    chain = prompt | llm
    ai_message = chain.invoke(
        {
            "law_name": law_name,
            "law_content": law_content,
        }
    )
    return ai_message

def judge_guilty_or_not(result, explanation, user_situation):
    """법 설명에 기반하여 유죄 여부 판단"""
    law_name = result.metadata
    law_content = result.page_content

    prompt = PromptTemplate.from_template("당신은 전문판사 입니다. {law_name}과 {law_content} 가 {user_situation}와 관련이 있는지 확인하고 관련이 없다면 '관련이 없는 법입니다'. 대부분은 관련이 없어. 사용자의 직업, 장소 등을 고려해서 객관적으로 판단해. 관련이 있다면 사용자가 {law_content}를 위반했는지 여부를 판단하고, 위반 여부와 이유를 2줄로 설명해. ")

    chain = prompt | llm
    ai_message = chain.invoke(
        {   "user_situation": user_situation,
            "law_name": law_name,
            "law_content": law_content,
        }
    )
    return ai_message

# 판례 관련 기능
def answer_precedent(user_situation):
    """사용자 상황을 판시사항 형식으로 요약하여 판례 검색"""
    prompt = PromptTemplate.from_template("너는 판사야. {user_situation}의 글을 판시사항 형식으로 3줄 미만으로 요약해서 줘")

    chain = prompt | llm
    ai_message = chain.invoke(
        {"user_situation": user_situation}
    )
    print('변환된 사용자의 질문:', ai_message.content)
    print("----------------------------------------------------------------------------------")

    # ai_message.content를 검색에 사용
    search_with_ai = retriever_precedent.get_relevant_documents(ai_message.content)
    return search_with_ai

def ai_summary(content):
    #"""주어진 판례를 쉽게 이해할 수 있도록 1줄로 요약"""
    prompt = PromptTemplate.from_template("주어진 {content}는 판시사항이야. 일반인이 이해하기 쉽도록 쉬운 말로 1줄로 요약해")

    chain = prompt | llm
    ai_message = chain.invoke(
        {"content": content}
    )
    return ai_message

def yes_or_no(situation, precedent):
  #판례가 user 상황과 맞는지 판단
    prompt = PromptTemplate.from_template("yes 혹은 no로 대답해. 주어진 {situation}가 {pre}와 관련이 있는지 확인하고 yes , no로 대답해줘. 대부분은 관련이 없어. ")
    ai_message = []


    chain = prompt | llm
    for pre in precedent:

        ai_message.append((chain.invoke(
            {"situation": situation, "pre": pre}  # 'situation'과 'pre'를 모두 포함해서 전달
        )).content)

    return ai_message

# 메인 서비스
def main():
    service_choice = 0
    while service_choice != "3":
      # 서비스 선택
      print("어떤 서비스를 원하십니까?")
      print("1. 법률찾기")
      print("2. 판례찾기")
      print("3. 종료")
      service_choice = input("1,2 또는 3을 선택해주세요: ")

      if service_choice == "1":
          # 법률찾기
          user_situation = input("당신이 처한 상황에 대해 말해주세요: ")
          print("--------------------------------------------------------")

          search_result = retriever_1.get_relevant_documents(user_situation)

          for result in search_result:
              start_time = time.time()
              print('법률:', result.metadata)
              print('내용:', result.page_content)
              explanations = explain_law(result)
              judgments = judge_guilty_or_not(result, explanations, user_situation)
              print('법 설명입니다:', explanations.content)
              print('범죄 성립 유무입니다:', judgments.content)
              end_time = time.time()
              elapsed_time = end_time - start_time
              print(f"대답을 받는데 걸린 시간: {elapsed_time:.4f} 초")
              print("--------------------------------------------------------")

      elif service_choice == "2":
          # 판례찾기
          user_situation = input("검색할 판례의 내용을 입력하세요: ")
          print("--------------------------------------------------------")

          # AI 기반 판례 검색
          ai_search = answer_precedent(user_situation)
          # 일반 검색
          normall_search = retriever_precedent.get_relevant_documents(user_situation)

          # 검색 결과 합치기 (중복 제거)
          all_search = ai_search + normall_search

          #yes or no 만들기
          all_content = []

          for i in range(len(all_search)):
            all_content.append(all_search[i].page_content)


          answer = yes_or_no(user_situation, all_content)

          unique_content = []
          unique_link = []
          unique_answer = []


          # 중복된 내용 및 링크 제거
          for num in range(len(all_search)):
              if  all_search[num].page_content not in unique_content:
                  unique_content.append(all_search[num].page_content)
                  unique_answer.append(answer[num])
              if all_search[num].metadata['link'] not in unique_link:
                  unique_link.append(all_search[num].metadata['link'])

          # 내용 출력
          for i in range(len(unique_content)):
              print('내용:', unique_content[i])
              print('요약:', ai_summary(unique_content[i]).content)
              print('관련있는가:', unique_answer[i])
              print('URL:', unique_link[i])
              print("--------------------------------------------------------")

      elif service_choice == "3":
          print("프로그램을 종료합니다.")
          break

      else:
          print("잘못된 입력입니다. 1 또는 2를 선택해주세요.")

# 서비스 실행
if __name__ == "__main__":
    main()
