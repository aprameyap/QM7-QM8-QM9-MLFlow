from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
np.random.seed(123)
import tensorflow as tf
tf.random.set_seed(123)
import deepchem as dc
import mlflow
import sys


tasks, datasets, transformers = dc.molnet.load_qm8()
train_dataset, valid_dataset, test_dataset = datasets


metric = [dc.metrics.Metric(dc.metrics.pearson_r2_score, mode="regression"), dc.metrics.Metric(dc.metrics.mean_absolute_error, mode="regression")]

batch_size = float(sys.argv[1]) if len(sys.argv) > 50 else 50
n_embedding = 20
n_distance = 51
distance_min = -1.
distance_max = 9.2
n_hidden = 15

with mlflow.start_run():
    model = dc.models.DTNNModel(
    len(tasks),
    n_embedding=n_embedding,
    n_hidden=n_hidden,
    n_distance=n_distance,
    distance_min=distance_min,
    distance_max=distance_max,
    output_activation=False,
    batch_size=batch_size,
    learning_rate=0.0001,
    use_queue=False,
    mode="regression")

    model.fit(train_dataset, nb_epoch=50)

    train_scores = model.evaluate(train_dataset, metric, transformers)
    valid_scores = model.evaluate(valid_dataset, metric, transformers)

    mlflow.log_param("Batch size", batch_size)
    mlflow.log_metric("MAE", valid_scores.get("mean_absolute_error"))
    mlflow.log_metric("Pearson R2", valid_scores.get("pearson_r2_score"))

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)
