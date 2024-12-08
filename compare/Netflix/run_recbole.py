import logging
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.quick_start.quick_start import load_data_and_model
from recbole.utils import get_model, get_trainer, init_logger
from recbole.utils.case_study import full_sort_topk
from replay.metrics import MAP, MRR, NDCG, HitRate, OfflineMetrics
from replay.metrics.torch_metrics_builder import metrics_to_df
from replay.preprocessing.filters import MinCountFilter
from rs_datasets import Netflix

save_outputs = False
LOAD_SAVED_MODEL = False
saved_model_path = ""

dataset_name = "interactions"
data_fraction = 0.4  # Taking this percent of last interactions as the data
log_dir = './compare/Netflix/logs_recbole'
data_dir = os.path.join(log_dir, dataset_name)
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

logging.basicConfig(level=logging.INFO, filename=os.path.join(log_dir, 'terminal_log.log'))
logger = logging.getLogger()
logger.setLevel(logging.INFO)

if not LOAD_SAVED_MODEL:
    df = Netflix("./netflix/")
    print(df.train.shape)
    interactions = df.train.loc[df.train["rating"] >= 3., ["user_id", "item_id", "timestamp"]]  # Only positive feedbacks
    print(interactions.shape)
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"]).astype(np.int64) + np.arange(0, interactions.shape[0])
    interactions = interactions.loc[interactions["timestamp"] > interactions["timestamp"].quantile(1 - data_fraction)].reset_index(drop=True)
    print(interactions.shape)
    interactions = MinCountFilter(3, "user_id").transform(interactions)
    print(interactions.shape)

    uid_encoder = {uid: i + 1 for i, uid in enumerate(set(interactions["user_id"]))}
    iid_encoder = {iid: i + 1 for i, iid in enumerate(set(interactions["item_id"]))}
    interactions["user_id"] = interactions["user_id"].map(uid_encoder)
    interactions["item_id"] = interactions["item_id"].map(iid_encoder)

    interactions = interactions.rename(columns={"user_id": "user_id:token",
                            "item_id": "item_id:token",
                            "timestamp": "timestamp:float"})
    interactions.to_csv(os.path.join(data_dir, f"{dataset_name}.inter"), sep="\t", index=False)
    del interactions
    del df

    start_time = time.time()

    config_dict = {
        'model': 'SASRec',
        'dataset': dataset_name,
        'data_path': log_dir,
        'epochs': 20,
        'train_batch_size': 512 * 64,
        'eval_batch_size': 512 * 4,
        'learning_rate': 0.001,
        'hidden_act': "relu",
        "shuffle": True,
        'train_neg_sample_args': None,
        'eval_args': {
            'split': {'LS': "valid_and_test"},
            'order': 'TO',
            'mode': 'uni100',
            'group_by_user': False
        },
        'metrics': ['Recall', 'NDCG', 'MAP'],
        'valid_metric': 'Recall@10',
        'topk': [1, 5, 10, 20],
        'seq_len': 50,
        'MAX_ITEM_LIST_LENGTH': 50,
        'hidden_size': 300,
        'num_heads': 2,
        'num_layers': 2,
        'inner_size': 300,
        'hidden_dropout_prob': 0.5,
        'attn_dropout_prob': 0.5,
        'layer_norm_eps': 1e-8,
        'initializer_range': 0.02,
        'loss_type': 'CE',
        'device': torch.device("cuda"),  # "cuda"
        'use_gpu': True,
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'TIME_FIELD': 'timestamp',
        'load_col': {
            'inter': ['user_id', 'item_id', 'timestamp']
        },
        'fields_in_same_space': [['user_id', 'item_id']],
        'convert_inter2seq': True
    }
    config = Config(model=config_dict['model'], dataset=config_dict['dataset'], config_dict=config_dict)
    init_logger(config)

    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)
    logger.addHandler(c_handler)
    # logger.info(config)

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    print(trainer.gpu_available)
    print(trainer.device)

    training_prep_time = time.time() - start_time
    with open(os.path.join(log_dir, "execution_time.log"), "w") as f:
        f.write(f"Training prep time: {training_prep_time} seconds\n")
    start_time = time.time()

    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    training_time = time.time() - start_time
    with open(os.path.join(log_dir, "execution_time.log"), "a") as f:
        f.write(f"Training time: {training_time} seconds\n")
        f.write(f"Inference prep time: {0.0} seconds\n")
else:
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=saved_model_path,
    )
    print(f"Loaded model: {model}")
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    trainer.resume_checkpoint(saved_model_path)

start_time = time.time()

test_metrics = trainer.evaluate(test_data, show_progress=True)

model.eval()
max_k = 100
batch_start = 0
recommendations = []
while batch_start < len(test_data.uid_list):
    print(f'Batch: {batch_start // config["eval_batch_size"] + 1} of {(len(test_data.uid_list) - 1) // config["eval_batch_size"] + 1}')
    uid_list = test_data.uid_list[batch_start:batch_start+config["eval_batch_size"]]
    topk_scores, topk_iid_list = full_sort_topk(uid_list, model, test_data, k=max_k, device=torch.device("cuda"))

    topk_scores = topk_scores.view(-1).cpu().detach().numpy()
    topk_iid_list = topk_iid_list.view(-1).cpu().detach().numpy()
    uid_list = np.repeat(uid_list, max_k)

    recommendations.append(pd.DataFrame({"query_id": uid_list,
                                         "item_id": topk_iid_list,
                                         "rating": topk_scores}))
    batch_start += config["eval_batch_size"]
print("Doing concatenation...")
recommendations = pd.concat(recommendations)
print("Removing extra...")
recommendations = recommendations.groupby("query_id").head(max_k).reset_index(drop=True)
print("Done recommendations")

true_list = []
test_interactions = test_data.dataset.inter_feat
test_interactions_df = pd.DataFrame({
    'user_id': test_interactions['user_id'].cpu().numpy(),
    'item_id': test_interactions['item_id'].cpu().numpy()
})
for user, items in test_interactions_df.groupby('user_id'):
    for item in items['item_id']:
        true_list.append((user, item))

true_df = pd.DataFrame(true_list, columns=["query_id", "item_id"])

inference_time = time.time() - start_time
with open(os.path.join(log_dir, "execution_time.log"), "a") as f:
    f.write(f"Inference time: {inference_time} seconds\n")

if save_outputs:
    recommendations.to_parquet(log_dir + "/test_predictions.parquet", engine="pyarrow")
    true_df.to_parquet(log_dir + "/true_df.parquet", engine="fastparquet")
    with open(log_dir + "/field2id_token.pickle", "wb") as f:
        pickle.dump(test_data.dataset.field2id_token, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(log_dir + "/field2token_id.pickle", "wb") as f:
        pickle.dump(test_data.dataset.field2token_id, f, protocol=pickle.HIGHEST_PROTOCOL)

TOPK = [1, 5, 10, 20, 100]
init_args = {
    "query_column": "query_id",
    "item_column": "item_id",
    "rating_column": "rating"
}

result_metrics = OfflineMetrics(
    [HitRate(TOPK), NDCG(TOPK), MAP(TOPK), MRR(TOPK)], **init_args
)(recommendations, true_df)

TF_METRICS = metrics_to_df(result_metrics)
print(TF_METRICS)
TF_METRICS.to_csv(os.path.join(log_dir, "metrics.csv"), index=False)
