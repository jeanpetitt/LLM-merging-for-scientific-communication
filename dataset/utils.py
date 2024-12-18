import requests
from typing import List


# get document from ask
def get_document_from_ask(question: List):
    pass

def get_comparison():
    headers = {
        'Content-Type': 'application/vnd.orkg.comparison.v2+json;charset=UTF-8',
        'Accept': 'application/vnd.orkg.comparison.v2+json',
        # "Authorization": "Bearer" + self._token.access_token
    }
    url = 'https://incubating.orkg.org/api/comparisons'
    try:
        response = requests.get(url, headers=headers)
        result = response.json()['content'][0]
        print(result)
    except Exception as e:
        raise e
    
if __name__ == "__main__":
    get_comparison()