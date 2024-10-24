# Bike-Chatbot_Azure-OpenAI
## 개요

Microsoft Azure OpenAI, Streamlit을 활용한 따릉이 정보 챗봇

## 구현 사항
### 대여소 목록 갱신
- **데이터 수신** - 서울시 열린데이터 광장에서 제공하는 서울시 공공자전거 실시간 대여정보 API에서 대여소이름, 위도, 경도 수집
- **위도, 경도 정보 처리** – 위에서 수집된 위도, 경도 값을 국토교통부 디지털트윈국토에서 제공하는 Geocoder API를 통해 주소로 변환
- **수집된 데이터 저장** – 대여소 이름, 변환된 주소를 SQLite를 사용해 데이터베이스에 저장

** 대여소 목록 갱신은 실시간으로 진행될 필요성이 적으므로 API 호출 횟수를 줄이기 위해 별도 코드 파일로 분리하여 필요시 진행하도록 함
### 실시간 이용 가능 자전거 대수 갱신
- **데이터 수신** -  서울시 열린데이터 광장에서 제공하는 서울시 공공자전거 실시간 대여정보 API에서 대여소와 각 대여소의 실시간 자전거 이용 가능 대수를 수신
- **수집된 데이터 저장** - DB에 이미 저장되어 있는 대여소 목록과 매칭하여 저장
### 사용자 질문 답변
- **따릉이 대여소 정보 제공**: 서울시의 공공 자전거 따릉이 대여소의 위치, 실시간 이용 가능 자전거 수 등을 답변
- **문장 유사도 분석**: 사용자의 입력과 대여소 정보를 비교하여 가장 유사한 대여소를 찾아 답변
- **일반적인 따릉이 정보 제공**: 따릉이 이용 방법, 요금, 반납 방법 등 따릉이와 관련된 일반 정보 답변
- **언어 모델 사용**: 사용자의 질문의도 파악 및 자연스러운 답변 가능


## 챗봇 작동 과정

1. **질문 입력**: 웹페이지에서 사용자 질문 입력
2. **질문 분류**:GPT 모델에 질문을 전달하여 사용자의 질문이 대여소 정보 관련 질문인지 일반적인 따릉이 관련 질문인지 분류
3. **질문 처리**
- 대여소 관련 질문일 경우, 서울시 공공자전거 실시간 대여정보 API로부터 최신 이용 가능 자전거 데이터를 가져와 데이터베이스를 업데이트, 사용자 입력과 데이터베이스에 저장된 대여소 정보 간의 유사도를 분석, 사용자의 질문과 가장 유사한 대여소 정보를 반환
- 일반 질문일 경우, Microsoft Azure 모델을 사용하여 답변 생성.
4. **결과 출력**: 사용자 질문에 대한 답변 출력

## 사용도구/기술

- **Python**: 개발언어
- **Streamlit**: 사용자 인터페이스
- **SQLite**: 자전거 대여소 정보를 저장하는 데이터베이스.
- **Sentence Transformers**: 문장 유사도 계산을 위한 사전 학습된 모델.
- **CountVectorizer**: 유사도 계산을 위한 n-그램 생성
- **Azure OpenAI**: 언어모델
- **Visual Studio Code**: 코드 작성
- **Windows**: 운영체제

## 파일 목록

### app_bike.py - 챗봇 코드 파일
### bike_stations.db - 대여소 데이터베이스
### update-station.py - 대여소 목록 업데이트 파일

## 실행결과 이미지
![스크린샷 2024-10-18 144733](https://github.com/user-attachments/assets/6e7cd5a0-fe2d-405b-b5c3-92426e3a45df)
![스크린샷 2024-10-18 144827](https://github.com/user-attachments/assets/f4d1d797-9b81-4441-acb2-f4169ddd90d5)
