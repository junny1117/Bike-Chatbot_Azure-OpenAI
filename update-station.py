import sqlite3
import requests
import json

def setup_database():
    conn = sqlite3.connect('bike_stations.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bike_stations (
            station_id TEXT PRIMARY KEY,
            station_name TEXT,
            address TEXT,
            bike_count INTEGER
        )
    ''')
    conn.commit()
    conn.close()

def save_bike_stations_to_db(bike_data):
    conn = sqlite3.connect('bike_stations.db')
    cursor = conn.cursor()
    for item in bike_data:
        cursor.execute('''
            INSERT OR REPLACE INTO bike_stations (station_id, station_name, address, bike_count)
            VALUES (?, ?, ?, ?)
        ''', (item['대여소ID'], item['대여소명'], item['주소'], 0))  
    conn.commit()
    conn.close()

def bike_info():
    urls = [
        'http://openapi.seoul.go.kr:8088/72436c414d6a756e3338596f6f5a50/json/bikeList/1/1000/',
        'http://openapi.seoul.go.kr:8088/72436c414d6a756e3338596f6f5a50/json/bikeList/1001/2000/',
        'http://openapi.seoul.go.kr:8088/72436c414d6a756e3338596f6f5a50/json/bikeList/2001/3000/'
    ]
    
    bike_data = []
    for url in urls:
        response = requests.get(url)
        content = response.content.decode('utf-8')
        data = json.loads(content)
        bike_data.extend(data['rentBikeStatus']['row'])  # Assuming the data structure
    
    return bike_data

def get_address_from_coordinates(lat, lon):
    geocode_url = f'http://api.vworld.kr/req/address?service=address&request=getAddress&version=2.0&crs=epsg:4326&point={lon},{lat}&format=json&type=parcel&key=78C2F437-0F01-3D96-B591-02D81C272AA9'
    response = requests.get(geocode_url)
    geocode_data = response.json()
    if geocode_data['response']['status'] == 'OK':
        address = geocode_data['response']['result'][0]['text']
        return address
    else:
        return None

def parse_bike_data(data):
    items = data
    parsed_data = []
    for item in items:
        lat = item.get('stationLatitude')
        lon = item.get('stationLongitude')
        address = get_address_from_coordinates(lat, lon)
        info = {
            '대여소명': item.get('stationName'),
            '대여소ID': item.get('stationId'),
            '주소': address
        }
        parsed_data.append(info)
    return parsed_data

setup_database()
bike_data = bike_info()
parsed_bike_data = parse_bike_data(bike_data)
save_bike_stations_to_db(parsed_bike_data)