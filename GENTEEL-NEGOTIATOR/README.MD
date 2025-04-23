
## Requirements

- `Python==3.8.0`
- `torch==1.10.0`
- `transformers==4.1.1`
- Download [Llama-3.1-8b-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct), and put it into the `meta-llama` folder

## Dataset


cd data/ConstructDataset
python construct_dataset.py --gpu 2
python politeness_extract.py

## Training, Testing and Evaluating

python rewards_main.py --direction forward --is_train True --is_test True --learning_rate 2e-5 --gpu 0

python rewards_main.py --direction backward --is_train True --is_test True --learning_rate 2e-5 --gpu 0

python main.py \
            --is_pretrain True \
            --is_with_pretrain True \
            --is_train True \
            --is_test True \
            --is_evaluate True \
            --is_evaluate_coher_pol_nego True \
            --save_evaluate_log save/log/evaluate.log \
            --save_test_log save/log/test.log \
		 --context_nego_reward_weigh 0.1 \
		 --future_nego_reward_weight 0.1 \
            --turn_reward_weight 0.1 \
            --conversation_reward_weight 0.1 \
            --context_reward_weight 1 \
            --future_reward_weight 1 \
            --save_method rewards \
            --results_file save/results/results.json \
            --gpu 0



