import requests


def request_for_sim_words(word):
    payload = {}
    headers = {}
    url = "https://pure-brushlands-42829.herokuapp.com/?word=[" + ','.join(word) + "]"
    response = requests.request("GET", url, headers=headers, data=payload)
    if response.status_code == 500:
        return [[] for w in word]
    return response.json()
