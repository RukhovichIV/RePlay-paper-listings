import logging
import os
import sys
import time

import lightning as L
import numpy as np
import pandas as pd
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from replay.data import (Dataset, FeatureHint, FeatureInfo, FeatureSchema,
                         FeatureSource, FeatureType)
from replay.data.nn import (SequenceTokenizer, SequentialDataset,
                            TensorFeatureInfo, TensorFeatureSource,
                            TensorSchema)
from replay.metrics import MAP, MRR, NDCG, HitRate, OfflineMetrics
from replay.metrics.torch_metrics_builder import metrics_to_df
from replay.models.nn.optimizer_utils import FatOptimizerFactory
from replay.models.nn.sequential import SasRec
from replay.models.nn.sequential.callbacks import (PandasPredictionCallback,
                                                   ValidationMetricsCallback)
from replay.models.nn.sequential.postprocessors import RemoveSeenItems
from replay.models.nn.sequential.sasrec import (SasRecPredictionDataset,
                                                SasRecTrainingDataset,
                                                SasRecValidationDataset)
from replay.preprocessing.filters import MinCountFilter
from replay.splitters import LastNSplitter
from rs_datasets import MovieLens
from torch.utils.data import DataLoader

save_outputs = False
log_dir = "compare/Movielens-20m/logs_replay"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(level=logging.INFO, handlers=[
    logging.FileHandler(os.path.join(log_dir, "terminal_log.log")),
    logging.StreamHandler(sys.stdout)
])
logger = logging.getLogger()

df = MovieLens("20m")
print(df.ratings.shape)
interactions = df.ratings.loc[df.ratings["rating"] >= 3., ["user_id", "item_id", "timestamp"]]  # Only positive feedbacks
print(interactions.shape)
interactions["timestamp"] = pd.to_datetime(interactions["timestamp"]).astype(np.int64) * 10**9 + np.arange(0, interactions.shape[0])
interactions = interactions.reset_index(drop=True)
interactions = MinCountFilter(3, "user_id").transform(interactions)
print(interactions.shape)

uid_encoder = {uid: i + 1 for i, uid in enumerate(set(interactions["user_id"]))}
iid_encoder = {iid: i + 1 for i, iid in enumerate(set(interactions["item_id"]))}
interactions["user_id"] = interactions["user_id"].map(uid_encoder)
interactions["item_id"] = interactions["item_id"].map(iid_encoder)

start_time = time.time()

splitter = LastNSplitter(
    N=1,
    divide_column="user_id",
    query_column="user_id",
    strategy="interactions",
)

raw_test_events, raw_test_gt = splitter.split(interactions)
if save_outputs:
    raw_test_events.to_parquet(log_dir + "/train_valid.parquet", engine="fastparquet")
    raw_test_gt.to_parquet(log_dir + "/true_df.parquet", engine="fastparquet")
raw_train_events, raw_validation_gt = splitter.split(raw_test_events)

def prepare_feature_schema(is_ground_truth: bool) -> FeatureSchema:
    base_features = FeatureSchema(
        [
            FeatureInfo(
                column="user_id",
                feature_hint=FeatureHint.QUERY_ID,
                feature_type=FeatureType.CATEGORICAL,
            ),
            FeatureInfo(
                column="item_id",
                feature_hint=FeatureHint.ITEM_ID,
                feature_type=FeatureType.CATEGORICAL,
            ),
        ]
    )
    if is_ground_truth:
        return base_features

    all_features = base_features + FeatureSchema(
        [
            FeatureInfo(
                column="timestamp",
                feature_type=FeatureType.NUMERICAL,
                feature_hint=FeatureHint.TIMESTAMP,
            ),
        ]
    )
    return all_features

user_features = pd.DataFrame({"user_id": interactions["user_id"].unique()})
item_features = pd.DataFrame({"item_id": interactions["item_id"].unique()})

train_dataset = Dataset(
    feature_schema=prepare_feature_schema(is_ground_truth=False),
    interactions=raw_train_events,
    check_consistency=True,
    categorical_encoded=False,
    item_features=item_features,
    query_features=user_features,
)
validation_dataset = Dataset(
    feature_schema=prepare_feature_schema(is_ground_truth=False),
    interactions=raw_train_events,
    check_consistency=True,
    categorical_encoded=False,
    item_features=item_features,
    query_features=user_features,
)
validation_gt = Dataset(
    feature_schema=prepare_feature_schema(is_ground_truth=True),
    interactions=raw_validation_gt,
    check_consistency=True,
    categorical_encoded=False,
    item_features=item_features,
    query_features=user_features,
)


test_dataset = Dataset(
    feature_schema=prepare_feature_schema(is_ground_truth=False),
    interactions=raw_test_events,
    check_consistency=True,
    categorical_encoded=False,
    item_features=item_features,
    query_features=user_features,
)
test_gt = Dataset(
    feature_schema=prepare_feature_schema(is_ground_truth=True),
    interactions=raw_test_gt,
    check_consistency=True,
    categorical_encoded=False,
    item_features=item_features,
    query_features=user_features,
)

ITEM_FEATURE_NAME = "item_id_seq"

tensor_schema = TensorSchema(
    TensorFeatureInfo(
        name=ITEM_FEATURE_NAME,
        feature_type=FeatureType.CATEGORICAL,
        is_seq=True,
        feature_hint=FeatureHint.ITEM_ID,
        feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, train_dataset.feature_schema.item_id_column)],
    )
)

tokenizer = SequenceTokenizer(tensor_schema, allow_collect_to_master=True)
tokenizer.fit(train_dataset)

sequential_train_dataset = tokenizer.transform(train_dataset)
sequential_test_dataset = tokenizer.transform(test_dataset)

sequential_val_gt = tokenizer.transform(validation_gt, [tensor_schema.item_id_feature_name])
sequential_val_dataset, sequential_val_gt = SequentialDataset.keep_common_query_ids(
    sequential_train_dataset, sequential_val_gt
)

sequential_test_gt = tokenizer.transform(test_gt, [tensor_schema.item_id_feature_name])
sequential_test_dataset, sequential_test_gt = SequentialDataset.keep_common_query_ids(
    sequential_test_dataset, sequential_test_gt
)

MAX_SEQ_LEN = 50
BATCH_SIZE = 2048
NUM_WORKERS = 10
NUM_EPOCHS = 20

model = SasRec(
    tensor_schema,
    block_count=2,
    head_count=2,
    max_seq_len=MAX_SEQ_LEN,
    hidden_size=300,
    dropout_rate=0.5,
    optimizer_factory=FatOptimizerFactory(learning_rate=0.001),
)

csv_logger = CSVLogger(save_dir=os.path.join(log_dir, ".logs/train"), name="SASRec_example")

checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(log_dir, ".checkpoints"),
    save_top_k=1,
    verbose=True,
    monitor="recall@10",
    mode="max",
)

validation_metrics_callback = ValidationMetricsCallback(
    metrics=["map", "ndcg", "recall"],
    ks=[1, 5, 10, 20],
    item_count=train_dataset.item_count,
    postprocessors=[RemoveSeenItems(sequential_val_dataset)]
)

trainer = L.Trainer(
    max_epochs=NUM_EPOCHS,
    callbacks=[checkpoint_callback, validation_metrics_callback],
    logger=csv_logger,
)

train_dataloader = DataLoader(
    dataset=SasRecTrainingDataset(
        sequential_train_dataset,
        max_sequence_length=MAX_SEQ_LEN,
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

validation_dataloader = DataLoader(
    dataset=SasRecValidationDataset(
        sequential_val_dataset,
        sequential_val_gt,
        sequential_train_dataset,
        max_sequence_length=MAX_SEQ_LEN,
    ),
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

training_prep_time = time.time() - start_time
with open(os.path.join(log_dir, "execution_time.log"), "w") as f:
    f.write(f"Training prep time: {training_prep_time} seconds\n")
start_time = time.time()

trainer.fit(
    model,
    train_dataloaders=train_dataloader,
    val_dataloaders=validation_dataloader,
)

training_time = time.time() - start_time
with open(os.path.join(log_dir, "execution_time.log"), "a") as f:
    f.write(f"Training time: {training_time} seconds\n")
start_time = time.time()

best_model = SasRec.load_from_checkpoint(checkpoint_callback.best_model_path)

prediction_dataloader = DataLoader(
    dataset=SasRecPredictionDataset(
        sequential_test_dataset,
        max_sequence_length=MAX_SEQ_LEN,
    ),
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

csv_logger = CSVLogger(save_dir=os.path.join(log_dir, ".logs/test"), name="SASRec_example")

TOPK = [1, 5, 10, 20, 100]

pandas_prediction_callback = PandasPredictionCallback(
    top_k=max(TOPK),
    query_column="user_id",
    item_column="item_id",
    rating_column="score",
    postprocessors=[RemoveSeenItems(sequential_test_dataset)],
)

predictor = L.Trainer(
    callbacks=[
        pandas_prediction_callback,
    ],
    logger=csv_logger,
    inference_mode=True
)

inference_prep_time = time.time() - start_time
with open(os.path.join(log_dir, "execution_time.log"), "a") as f:
    f.write(f"Inference prep time: {inference_prep_time} seconds\n")
start_time = time.time()

predictor.predict(best_model, dataloaders=prediction_dataloader, return_predictions=False)

pandas_res = pandas_prediction_callback.get_result()
recommendations = tokenizer.query_and_item_id_encoder.inverse_transform(pandas_res)

inference_time = time.time() - start_time
with open(os.path.join(log_dir, "execution_time.log"), "a") as f:
    f.write(f"Inference time: {inference_time} seconds\n")

if save_outputs:
    recommendations.to_parquet(log_dir + "/test_predictions.parquet", engine="pyarrow")

init_args = {"query_column": "user_id", "rating_column": "score"}

result_metrics = OfflineMetrics(
    [HitRate(TOPK), NDCG(TOPK), MAP(TOPK), MRR(TOPK)], **init_args
)(recommendations, raw_test_gt)

TORCH_METRICS = metrics_to_df(result_metrics)
print(TORCH_METRICS)
metrics_path = os.path.join(log_dir, f"metrics.csv")
TORCH_METRICS.to_csv(metrics_path, index=False)
