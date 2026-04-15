import requests
import json

# API endpoint
url = "http://222.90.211.46:20080/api/v1/dictionary/test"

# API key
api_key = "ragflow-7VR4ItrAzNQUIljrZzR6mGGwQERNlHgAqtY3oQDXdNQ"

# Headers
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Test sentences containing "线路串抗"
test_sentences = [
    "线路串抗整个句子测试",
    "这个线路串抗设备运行正常",
    "线路串抗是一种重要的电力设备",
    "我们需要安装新的线路串抗",
    "线路串抗的作用是什么？"
]

# Test multiple times for each sentence
for sentence in test_sentences:
    print(f"\n测试句子: {sentence}")
    for i in range(3):
        print(f"第{i+1}次测试:")
        try:
            response = requests.post(url, headers=headers, json={"text": sentence})
            print(f"状态码: {response.status_code}")
            result = response.json()
            print(f"完整响应: {json.dumps(result, ensure_ascii=False, indent=2)}")
        except Exception as e:
            print(f"错误: {str(e)}")
