import re
from sklearn.metrics import f1_score
from bert_score import score
import numpy as np

"""
Requires the packages bert-score and scikit-learn to be installed:

pip install bert-score
pip install scikit-learn

"""

# Case and leading/trailing whitespace insensitive
def exact_match(answer, expected_answer):
    return 1.0 if answer.strip().lower() == expected_answer.strip().lower() else 0.0

def exact_matches(answers, expected_answers):
    return [exact_match(answer, expected_answer) for answer, expected_answer in zip(answers, expected_answers)]

# Case and leading/trailing whitespace insensitive
def f1_token_level(answer, expected_answer):
    answer_tokens = answer.strip().lower().split()
    expected_tokens = expected_answer.strip().lower().split()

    common_tokens = set(answer_tokens) & set(expected_tokens)    
    precision = len(common_tokens) / len(answer_tokens)
    recall = len(common_tokens) / len(expected_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def f1_token_levels(answers, expected_answers):
    return [f1_token_level(answer, expected_answer) for answer, expected_answer in zip(answers, expected_answers)]

def bert_score(answer, expected_answer):
    P, R, F1 = score([answer], [expected_answer], model_type="microsoft/deberta-xlarge-mnli", lang="en")
    return F1.item()


def bert_scores(answers, expected_answers):
    P, R, F1 = score(answers, expected_answers, model_type="microsoft/deberta-xlarge-mnli", lang="en")
    return F1.tolist()

if __name__ == '__main__':
    
    expected_answer = "This is an extraordinarily correct sentence."
    answers = [
        "This is an extraordinarily correct sentence.",
        "This is an extremely correct sentence.",
        "This sentence is mostly correct.",
        "I sentence you to correct yourself.", 
        "I prefer wine over flowers."
    ]
    """
    for label, answer in answers:
        evaluation = evaluate_all_metrics(answer, expected_answer)
        print(f"Evaluation Results for {label}:")
        for metric, eval_score in evaluation.items():
            print(f"  {metric}: {eval_score}")
        print()
    """
    print(exact_matches([expected_answer] * 5, answers))
    print(f1_token_levels([expected_answer] * 5, answers))
    print(bert_scores([expected_answer] * 5, answers))



