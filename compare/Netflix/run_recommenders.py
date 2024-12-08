import logging
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from recommenders.models.sasrec.model import SASREC
from recommenders.models.sasrec.sampler import WarpSampler
from recommenders.models.sasrec.util import SASRecDataSet
from replay.metrics import MAP, MRR, NDCG, HitRate, OfflineMetrics
from replay.metrics.torch_metrics_builder import metrics_to_df
from replay.preprocessing.filters import MinCountFilter
from rs_datasets import Netflix
from tqdm import tqdm

save_outputs = False
log_dir = './compare/Netflix/logs_recommenders'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logging.basicConfig(level=logging.INFO, filename=os.path.join(log_dir, 'terminal_log.log'))
logger = logging.getLogger()
logger.setLevel(logging.INFO)

data_fraction = 0.4  # Taking this percent of last interactions as the data
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

interactions.to_csv(log_dir + "/raw_dataset.csv", index=False, sep=' ', header=False)

start_time = time.time()

dataset = SASRecDataSet(filename=log_dir + "/raw_dataset.csv", col_sep=' ')
dataset.split()

num_epochs = 20
batch_size = 2048
lr = 0.001
maxlen = 50
num_blocks = 2
hidden_units = 300
num_heads = 2
dropout_rate = 0.5
l2_emb = 0.0
num_neg_test = len(iid_encoder) - 1
max_k = 100

device = tf.device('/GPU:0')

with device:
    model = SASREC(
        item_num=dataset.itemnum,
        seq_max_len=maxlen,
        num_blocks=num_blocks,
        embedding_dim=hidden_units,
        attention_dim=hidden_units,
        attention_num_heads=num_heads,
        dropout_rate=dropout_rate,
        conv_dims=[hidden_units, hidden_units],
        l2_reg=l2_emb,
        num_neg_test=num_neg_test
    )

    sampler = WarpSampler(dataset.user_train, dataset.usernum, dataset.itemnum, batch_size=batch_size, maxlen=maxlen, n_workers=3)

    training_prep_time = time.time() - start_time
    with open(os.path.join(log_dir, "execution_time.log"), "w") as f:
        f.write(f"Training prep time: {training_prep_time} seconds\n")
    start_time = time.time()

    t_test = model.train(dataset, sampler, num_epochs=num_epochs, batch_size=batch_size, lr=lr, val_epoch=num_epochs + 1)
    
    training_time = time.time() - start_time
    with open(os.path.join(log_dir, "execution_time.log"), "a") as f:
        f.write(f"Training time: {training_time} seconds\n")
    start_time = time.time()

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)

def get_predictions_for_test_users(model, dataset, maxlen):
    recommendations = []
    with device:
        item_idx = np.arange(1, dataset.itemnum + 1)
        for user in tqdm(dataset.user_test.keys(), desc="Predicting", ncols=70):
            input_seq = np.zeros([maxlen], dtype=np.int32)
            interaction_list = dataset.user_train[user]
            if user in dataset.user_valid and dataset.user_valid[user]:
                interaction_list += [dataset.user_valid[user][0]]

            interaction_list = interaction_list[-maxlen:]
            input_seq[-len(interaction_list):] = interaction_list

            inputs = {
                "user": np.expand_dims(np.array([user]), axis=0),
                "input_seq": np.expand_dims(input_seq, axis=0),
                "candidate": np.expand_dims(item_idx, axis=0)
            }
            model.num_neg_test = dataset.itemnum - 1

            preds = model.predict(inputs)[0].numpy()
            seen_items = np.array(interaction_list) - 1
            preds[seen_items] = -np.inf
            probabilities = softmax(preds)

            nan_count = np.isnan(probabilities).sum()
            if nan_count + max_k > probabilities.shape[0]:
                print(f"Strange user: {user}. Got {nan_count} NaNs out of {probabilities.shape[0]}")
                continue
            if nan_count > 0:
                indices = np.argpartition(probabilities, [-max_k - nan_count, -nan_count])[-max_k - nan_count : -nan_count]
            else:
                indices = np.argpartition(probabilities, [-max_k])[-max_k:]
            indices = indices[np.argsort(-probabilities[indices])]
            values = probabilities[indices]
            indices += 1

            recommendations.append(pd.DataFrame({"query_id": np.repeat(user, values.shape[0]),
                                                 "item_id": indices,
                                                 "rating": values}))
    return pd.concat(recommendations)

inference_prep_time = time.time() - start_time
with open(os.path.join(log_dir, "execution_time.log"), "a") as f:
    f.write(f"Inference prep time: {inference_prep_time} seconds\n")
start_time = time.time()

test_predictions = get_predictions_for_test_users(model, dataset, maxlen)

inference_time = time.time() - start_time
with open(os.path.join(log_dir, "execution_time.log"), "a") as f:
    f.write(f"Inference time: {inference_time} seconds\n")

true_list = []
for user, items in dataset.user_test.items():
    for item in items:
        true_list.append((user, item))

true_df = pd.DataFrame(true_list, columns=["query_id", "item_id"])
if save_outputs:
    test_predictions.to_parquet(log_dir + "/test_predictions.parquet", engine="fastparquet")
    true_df.to_parquet(log_dir + "/true_df.parquet", engine="fastparquet")

TOPK = [1, 5, 10, 20, 100]
init_args = {
    "query_column": "query_id",
    "item_column": "item_id",
    "rating_column": "rating"
}

result_metrics = OfflineMetrics(
    [HitRate(TOPK), NDCG(TOPK), MAP(TOPK), MRR(TOPK)], **init_args
)(test_predictions, true_df)
TF_METRICS = metrics_to_df(result_metrics)
print(TF_METRICS)
TF_METRICS.to_csv(os.path.join(log_dir, "metrics.csv"), index=False)
