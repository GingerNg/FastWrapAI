import requests

resp = requests.post("http://XX.XX.XX.XX:8077/user/register", json={"name": "dede"}, headers={"Authorization": "12121"})

print(resp.json())