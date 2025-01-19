from rouge_score import rouge_scorer
import json
from typing import Literal

TYPE_GENERATION = Literal['title', 'summary', 'both']

def compute_rouge_average(
    reference_json_path, 
    candidate_json_path,
    typeG: TYPE_GENERATION
    ):
    """
    Compute the average ROUGE scores between two JSONL files by matching lines based on IDs.

    Args:
        reference_json_path (str): Path to the JSONL file containing reference data.
        candidate_json_path (str): Path to the JSONL file containing candidate data.
        typeG (TYPE_GENERATION): type

    Returns:
        dict: The average ROUGE scores for rouge1, rouge2, and rougeL.
    """
    # Initialize the ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Load the JSONL files
    with open(reference_json_path, 'r', encoding='utf-8') as ref_file, \
         open(candidate_json_path, 'r', encoding='utf-8') as cand_file:
        
        # Convert to dictionaries {id: data}
        reference_data = {json.loads(line)['id']: json.loads(line) for line in ref_file}
        candidate_data = {json.loads(line)['id']: json.loads(line) for line in cand_file}
    
    # Check for common IDs
    common_ids = set(reference_data.keys()) & set(candidate_data.keys())
    if not common_ids:
        raise ValueError("No common IDs found between the two files.")
    
    # Variables to accumulate scores
    total_rouge1 = 0.0
    total_rouge2 = 0.0
    total_rougeL = 0.0
    count = 0
    
    # Compute ROUGE scores for each common ID
    for id_ in common_ids:
        ref = reference_data[id_]
        cand = candidate_data[id_]
        
        # Prepare the texts
        if typeG == "title":
            reference_text = f"{ref['answer']['title']}"
            candidate_text = f"{cand['title']}"
        elif typeG == "summary":
            reference_text = f"{ref['answer']['summary']}"
            candidate_text = f"{cand['summary']}"
        elif typeG == "both":
            reference_text = f"{ref['answer']['title']} {ref['answer']['summary']}"
            candidate_text = f"{cand['title']} {cand['summary']}"
            
        else:
            raise ValueError("You should provide a correct typeG(title, summary, both)")
        
        
        # Calculate the scores for this pair
        scores = scorer.score(reference_text, candidate_text)
        
        # Accumulate the scores
        total_rouge1 += scores['rouge1'].fmeasure
        total_rouge2 += scores['rouge2'].fmeasure
        total_rougeL += scores['rougeL'].fmeasure
        count += 1
    
    # Calculate the averages
    return {
        "rouge1": total_rouge1 / count,
        "rouge2": total_rouge2 / count,
        "rougeL": total_rougeL / count,
    }




if __name__ == "__main__":
    # # Example usage
    reference_path = "test.jsonl"
    candidate_path = "results/few_shot_openai_output.jsonl"
    rouge_scores = compute_rouge_average(
        reference_path,
        candidate_path,
        typeG="both"
    )
    print(rouge_scores)

    