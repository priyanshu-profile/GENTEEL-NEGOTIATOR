import json
import pickle
from collections import Counter, defaultdict

if __name__ == "__main__":
    politeness_dict = json.load(open("../politeness.json",encoding="utf-8"))
    [data_train, data_dev, data_test] = pickle.load(open("../dataset_preproc_politeness_cache.p","rb"))
    politeness_labels = list()
    for data in [data_train, data_dev, data_test]:
        for uttr_cs, next_uttr_cs in zip(data["utt_cs"], data["next_uttr_cs"]):
            for uttr_reaction in uttr_cs["xPol"]:
                if len(uttr_reaction) == 1:
                    politeness_labels.extend(uttr_reaction)
            for next_uttr_reaction in next_uttr_cs["xPol"]:
                if len(next_uttr_reaction) == 1:
                    politeness_labels.extend(next_uttr_reaction)
    politeness_counter = dict(Counter(politeness_labels))
    politeness_counter = sorted(politeness_counter.items(), key=lambda x: x[1], reverse=True) 
    politeness_counter = [(politeness, cnt, politeness_dict[politeness][0]) for politeness, cnt in politeness_counter if politeness in politeness_dict and cnt > 40] 
    politeness_statistic = defaultdict(dict)
    for idx, (politeness, cnt, score) in enumerate(politeness_counter):
        politeness = "positive" if score >= 0.5 else "negative"
        politeness_statistic[politeness] = {
            "idx": idx+1,
            "politeness": politeness_polarity,
            "count": cnt,
            "pol": politeness_dict[politeness],
        }
    with open("../politeness_statistic.json", "w", encoding="utf-8") as f:
        json.dump(politeness_statistic, f, ensure_ascii=False, indent=2)
    
    print("Save Extracted politeness Label!")
