import requests
from typing import List
import json

# get document from ask
def get_document_from_ask(question: List):
    pass

def retrieve_comparison(id):
    headers = {
        'Content-Type': 'application/vnd.orkg.comparison.v2+json;charset=UTF-8',
        'Accept': 'application/vnd.orkg.comparison.v2+json',
        # "Authorization": "Bearer" + self._token.access_token
    }
    url = f'https://orkg.org/api/comparisons/{id}'
    try:
        return _extracted_from_retrieve_comparison_9(url, headers)
    except Exception as e:
        raise e


# TODO Rename this here and in `retrieve_comparison`
def _extracted_from_retrieve_comparison_9(url, headers):
    response = requests.get(url, headers=headers)
    result = response.json()
    # print(result)
    criteria = "\n".join([f"{i+1}. {item['label']}" for i, item in enumerate(result['data']['predicates'])])
    contributions = "\n".join([f"{i+1}-{item['label']} extracted from paper entitled: {item['paper_label']}" for i, item in enumerate(result['data']['contributions'])])
    return {
        'instruction': f"""Here are {len(result['contributions'])} contributions to analyze and compare:\n## Criteria:\n{criteria} \n## Contributions:\n {contributions} \n## Question:\nProvide a concise title and summary of this comparison based on the listed criteria and contributions.""",
        'answer': {
            'title': result['title'],
            'summary': result['description'],
        },
        'id': result['id']
    }

def get_comparisons(comparison_ids=None):
    if not comparison_ids:
        url = 'https://orkg.org/api/comparisons'
        headers = {
            'Content-Type': 'application/vnd.orkg.comparison.v2+json;charset=UTF-8',
            'Accept': 'application/vnd.orkg.comparison.v2+json',
            # "Authorization": "Bearer" + self._token.access_token
        }
        try:
            _extracted_from_get_comparisons_10(url, headers)
        except Exception as e:
            raise e
    else:
        results = []
        for id in comparison_ids:
            print(id)
            results.append(retrieve_comparison(id))
        with open('test.jsonl', 'w') as f:
            for item in results:
                json.dump(item, f)
                f.write('\n')


# TODO Rename this here and in `get_comparisons`
def _extracted_from_get_comparisons_10(url, headers):
    response = requests.get(url, headers=headers, params={"page": 100})
    results = response.json()['content']
    # print(results)
    dataset = {}
    data = []
    for result in results:
                # if result['created_by'] == "07a4c04e-f0ed-407e-a097-b676d5228a40":
            criteria = "\n".join([f"{i+1}. {item['label']}" for i, item in enumerate(result['data']['predicates'])])
            contributions = "\n".join([f"{i+1}-{item['label']} extracted from paper entitled: {item['paper_label']}" for i, item in enumerate(result['data']['contributions'])])
            dataset['instruction'] = f"""Here are {len(result['contributions'])} contributions to analyze and compare:
            ## Criteria:\n{criteria} \n## Contributions:\n {contributions} \n## Question:\nProvide a concise title and summary of this comparison based on the listed criteria and contributions."""
            dataset['answer'] = {'title': result['title'], 'summary': result['description']}
            dataset['id'] = result['id']
            data.append(dataset)
    print(data)
    print(len(results))
    with open('test.jsonl', 'w') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')
    
if __name__ == "__main__":
    comparison_ids = [
        "R36099", "R8342", "R739481", "R700971", "R679266", "R259184",
        "R267027", "R269002", "R272021", "R272059", "R272079", "R288078",
        "R576864", "R576876", "R600534", "R601687", "R604322", "R609853",
        "R642230", "R642232", "R642266", "R646599", "R646606", "R646615",
        "R655964", "R656485", "R656502", "R657613", "R657615", "R657617",
        "R657623", "R657627", "R657627", "R657630", "R657632", "R642234",
        "R691972", "R653209", "R184018", "R193988", "R197375", "R198562",
        "R206242", "R206258", "R213085", "R217404", "R217418", "R217421",
        "R221864", "R222164", "R655553", "R655555"]
    get_comparisons(comparison_ids)
    # retrieve_comparison("R655553")