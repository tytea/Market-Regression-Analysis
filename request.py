import requests

url = 'http://localhost:5000/api'

price = int(input("Price: "))


r = requests.post(url,json={'exp':price,})
print(r.json())