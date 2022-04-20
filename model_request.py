import requests


def request_for_sim_words(word):
    payload = {}
    headers = {}
    url = "https://pure-brushlands-42829.herokuapp.com/?word=[" + ','.join(word) + "]"
    response = requests.request("GET", url, headers=headers, data=payload)
    return response.json()
