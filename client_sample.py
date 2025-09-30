import requests

# 创建 session 并禁用代理
session = requests.Session()
session.trust_env = False  

# 中文转录接口调用样例
files_zh = {'file': open('audios/zh_sample.mp3', 'rb')}
data_zh = {'language_code': 'zh'}
response_zh = session.post('http://localhost:8000/transcribe', files=files_zh, data=data_zh)
if response_zh.status_code == 200:
    print(response_zh.json())
else:
    print(f"Error: {response_zh.status_code}, {response_zh.text}")

# # 英文转录接口调用样例
# files_en = {'file': open('audios/en_sample.mp3', 'rb')}
# data_en = {'language_code': 'en'}
# response_en = session.post('http://localhost:8000/transcribe', files=files_en, data=data_en)
# if response_en.status_code == 200:
#     print(response_en.json())
# else:
#     print(f"Error: {response_en.status_code}, {response_en.text}")