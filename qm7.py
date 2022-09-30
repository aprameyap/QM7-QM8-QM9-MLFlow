import deepchem as dc
from deepchem.molnet.load_function.qm7_datasets import load_qm7

 #loading qm7

(tasks, datasets, transformers) = dc.molnet.load_qm7(move_mean=False)
(train_dataset, valid_dataset, test_dataset) = datasets

regression_metric = dc.metrics.Metric(dc.metrics.mean_absolute_error, mode="regression")
