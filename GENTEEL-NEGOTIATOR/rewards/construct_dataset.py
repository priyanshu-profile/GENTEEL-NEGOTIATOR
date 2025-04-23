import os
import json
import pickle
import random
from tqdm import tqdm

if __name__ == "__main__":
    random.seed(42)
    [data_train, data_val, data_test] = pickle.load(open("../data/dataset_preproc.p","rb"))
    total_target = data_train["target"] + data_val["target"] + data_test["target"]
    total_target_pos_kts = data_train["target_positive_kts"] + data_val["target_positive_kts"] + data_test["target_positive_kts"]
    total_target_neg_kts = data_train["target_negative_kts"] + data_val["target_negative_kts"] + data_test["target_negative_kts"]
    total_startegy = data_train["strategy"] + data_val["strategy"] + data_test["strategy"]
    assert len(total_target) == len(total_target_neg_kts) == len(total_target_pos_kts) == len(total_startegy) 
    
    for dataset in [data_train, data_val, data_test]:
        dataset["label"] = [1] * len(dataset["context"])
        sample_dataset_idx = random.sample(range(len(dataset["context"])), int(len(dataset["context"])/2))
        assert len(sample_dataset_idx) == len(set(sample_dataset_idx))
        for idx in tqdm(sample_dataset_idx):
            sample_idx = random.sample(range(len(total_target)), 1)[0]
            while " ".join(total_target[sample_idx]).strip() == " ".join(dataset["target"][idx]).strip():
                sample_idx = random.sample(range(len(total_target)), 1)[0]
            dataset["target"][idx] = total_target[sample_idx]
            dataset["target_positive_kts"][idx] = total_target_pos_kts[sample_idx]
            dataset["target_negative_kts"][idx] = total_target_neg_kts[sample_idx]
            dataset["strategy"][idx] = total_startegy[sample_idx]
            dataset["label"][idx] = 0
    
    with open("./rewardsdataset.pkl", "wb") as f:
        pickle.dump([data_train, data_val, data_test], f)
    
    print("Done!")