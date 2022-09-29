from sklearn import datasets, linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score
import os
import mlflow
import deepchem as dc
from deepchem.molnet.load_function.qm7_datasets import load_qm7


(tasks, datasets, transformers) = dc.molnet.load_qm7(move_mean=False)
(train_dataset, valid_dataset, test_dataset) = datasets

regression_metric = dc.metrics.Metric(dc.metrics.mean_absolute_error, mode="regression")
