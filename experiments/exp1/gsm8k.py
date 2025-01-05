import datasets
from dataclasses import dataclass

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

@dataclass
class GSM8KExample:
    question: str
    answer_str: str
    answer_int: int

def gsm8k_processed(split='test'):
    ds = datasets.load_dataset("openai/gsm8k", "main")
    data = ds[split]
    results = []
    for d in ds[split]:
        answer_str = d['answer'].split('####')[1].strip().replace(",", "")
        assert is_number(answer_str)
        answer_int = int(float(answer_str))
        results.append(GSM8KExample(
            question=d['question'],
            answer_str=answer_str,
            answer_int=answer_int,
        ))
    return results