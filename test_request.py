"""
Script to test a single GET request to the exposed API that hosts the model.
"""
import requests


URL = "http://10.192.65.253:5000/"

input = {'tweet':'A photo shows a 19-year-old vaccine for canine coronavirus that could be used to prevent the new coronavirus causing COVID-19.'}
print(input)

r = requests.get(url = URL, params = input)
data = r.json()
print(data)

input = {'tweet':'Households should have ""required"" medical kits with certain items and equipment to treat the different stages of COVID-19.'}
print(input)

r = requests.get(url = URL, params = input)
data = r.json()
print(data)