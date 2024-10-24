import requests
import json
import sqlite3
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import CountVectorizer

endpoint = "https://st035-openai.openai.azure.com/"
deployment_name = "st035-gpt4o"
api_key = "51609ef8e82e4614b5257b34c79f4a39"

headers = {
    "Content-Type": "application/json",
    "api-key": api_key
}

def request_chatgpt(user_message, system_message):
    data = {
        "messages": [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        "temperature": 0.5,
        "top_p": 0.95,
        "max_tokens": 1000
    }

    response = requests.post(f"{endpoint}openai/deployments/{deployment_name}/chat/completions?api-version=2024-02-15-preview",
                headers=headers,
                json=data)
    
    response_json = response.json()
    return response_json['choices'][0]['message']['content']


def get_bike_info():
    urls = [
        'http://openapi.seoul.go.kr:8088/72436c414d6a756e3338596f6f5a50/json/bikeList/1/1000/',
        'http://openapi.seoul.go.kr:8088/72436c414d6a756e3338596f6f5a50/json/bikeList/1001/2000/',
        'http://openapi.seoul.go.kr:8088/72436c414d6a756e3338596f6f5a50/json/bikeList/2001/3000/'
    ]
    bike_data = []
    for url in urls:
        response = requests.get(url)
        data = response.json()
        bike_data.extend(data['rentBikeStatus']['row'])
    return bike_data

def update_bike_counts():
    conn = sqlite3.connect('bike_stations.db')
    cursor = conn.cursor()
    
    bike_data = get_bike_info()
    for item in bike_data:
        station_id = item['stationId']
        bike_count = item['parkingBikeTotCnt']
        cursor.execute('''
            UPDATE bike_stations
            SET bike_count = ?
            WHERE station_id = ?
        ''', (bike_count, station_id))
    
    conn.commit()
    conn.close()

def generate_ngrams(text, n=2):
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
    ngrams = vectorizer.fit_transform([text])
    ngram_features = vectorizer.get_feature_names_out()
    return ngram_features

def calculate_similarity(user_input, station_name, station_address):
    user_ngrams = generate_ngrams(user_input)
    station_ngrams = generate_ngrams(f"{station_name} {station_address}")
    
    common_ngrams = set(user_ngrams).intersection(set(station_ngrams))
    similarity = len(common_ngrams) / max(len(user_ngrams), len(station_ngrams))
    return similarity


def get_bike_station_info(station_name):
    conn = sqlite3.connect('bike_stations.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT station_name, address, bike_count
        FROM bike_stations
        WHERE station_name = ?
    ''', (station_name,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return {
            '대여소명': result[0],
            '주소': result[1],
            '자전거수': result[2]
        }
    else:
        return None


def extract_keywords(user_message):
    keywords = user_message.split()
    filtered_keywords = [word for word in keywords if word not in ["대여소", "자전거"]]
    return filtered_keywords

def find_most_similar_stations(user_message, top_k=50, similarity_threshold=0.1):
    conn = sqlite3.connect('bike_stations.db')
    cursor = conn.cursor()
    cursor.execute('SELECT station_name, address FROM bike_stations')
    stations = cursor.fetchall()
    conn.close()
    
    keywords = extract_keywords(user_message) 
    
    similarities = []
    
    for station_name, station_address in stations:
        if station_name is None:
            station_name = ""
        if station_address is None:
            station_address = ""
        
        if any(keyword in (station_name + station_address) for keyword in keywords):
            similarity = calculate_similarity(user_message, station_name, station_address)
            if similarity >= similarity_threshold:
                similarities.append((station_name, station_address, similarity))
    
    similarities = sorted(similarities, key=lambda x: x[2], reverse=True)[:top_k]
    
    most_similar_stations = []
    for station_name, station_address, _ in similarities:
        station_info = get_bike_station_info(station_name)
        if station_info:
            most_similar_stations.append(station_info)
    
    return most_similar_stations


def classify_question(user_message):
    system_message = (
        "사용자가 서울시 공공 자전거 시스템인 따릉이에 대해 질문합니다. 반드시 따릉이 관련 질문에만 대답하세요."
        "다음 질문이 '대여소 정보'와 관련된 질문인지 아니면 '일반적인 따릉이 정보'와 관련된 질문인지 분류하세요. "
        "'대여소 정보'는 따릉이 대여소 위치, 자전거 수 등과 관련된 질문입니다. "
        "'일반적인 따릉이 정보'는 따릉이 이용 방법, 요금, 반납 방법 등 대여소 정보가 아닌 질문입니다."
    )
    classification_prompt = f"질문: {user_message}"
    
    classification_result = request_chatgpt(classification_prompt, system_message)
    
    if "대여소 정보" in classification_result:
        return "대여소 관련 질문"
    elif "일반적인 따릉이 정보" in classification_result:
        return "일반 질문"
    else:
        return "분류 실패"

def handle_user_question(user_message):
    question_type = classify_question(user_message)
    
    if question_type == "대여소 관련 질문":
        station_infos = find_most_similar_stations(user_message)
        if station_infos:
            return "\n\n".join([f"대여소명: {info['대여소명']},\n주소: {info['주소']},\n자전거수: {info['자전거수']}" for info in station_infos if info])
        else:
            return "죄송합니다, 해당 대여소 정보를 찾을 수 없습니다."
    
    elif question_type == "일반 질문":
        system_message = "서울특별시 따릉이와 관련된 질문에 대해 답변해 주세요. 반드시 따릉이 관련 질문에만 대답하세요."
        gpt_response = request_chatgpt(user_message, system_message)
        return gpt_response
    
    else:
        return "저는 따릉이 관련 질문에만 답변할 수 있습니다."

def main():
    st.title("따릉이 챗봇")
    user_message = st.text_input("질문을 입력하세요:", "")
    
    if st.button("제출"):
        # 자전거 수 업데이트
        update_bike_counts()
        
        response_message = handle_user_question(user_message)
        
        st.markdown(response_message)

if __name__ == "__main__":
    main()
