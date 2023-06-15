import jsonlines
from pathlib import Path
import pandas as pd


models = ["VisualQuestionAnswering", "flan-alpaca-xxl", "flan-t5-xxl", "stable-vicuna-13B-HF"]

path = Path("./subjective")

data = {}
for model in models:
    lst = []
    filename = path / model / "coherence.jsonl"
    with jsonlines.open(filename) as coh:
        filename = path / model / "relevance.jsonl"
        with jsonlines.open(filename) as rel:
            for c, r in zip(coh, rel):
                lst.append( (c["Score"], r["Score"]) )
    data[model[5:10] + "col,rel"] = lst

pd.set_option('display.max_rows', None)

print(pd.DataFrame(data))
