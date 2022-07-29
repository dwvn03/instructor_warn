import requests

def warningPing():
    url = "http://localhost:8080/warn/"
    requests.get(url)