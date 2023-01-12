import requests

URL = "http://10.192.65.253:5000/"

# defining a params dict for the parameters to be sent to the API
PARAMS = {'tweet':'A photo shows a 19-year-old vaccine for canine coronavirus that could be used to prevent the new coronavirus causing COVID-19.'}
  
# sending get request and saving the response as response object
r = requests.get(url = URL, params = PARAMS)
data = r.json()
print(data)