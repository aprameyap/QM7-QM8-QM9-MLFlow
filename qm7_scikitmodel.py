from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
np.random.seed(123)
import tensorflow as tf
tf.random.set_seed(123)
import deepchem as dc
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import sys

featurizer = dc.feat.CoulombMatrix(max_atoms = 50)

tasks, datasets, transformers = dc.molnet.load_qm7(
    featurizer= featurizer, splitter='stratified', move_mean=False, reload=False)

train, valid, test = datasets

regression_metric = dc.metrics.Metric(
    dc.metrics.mean_absolute_error, mode="regression")


alpha = float(sys.argv[1]) if len(sys.argv) < 1 else 5e-4
gamma = float(sys.argv[2]) if len(sys.argv) < 1 else 0.008

with mlflow.start_run():
    def model_builder(model_dir):
        sklearn_model = KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma)
        return dc.models.SklearnModel(sklearn_model, model_dir)
        
model = dc.models.SingletaskToMultitask(tasks, model_builder)

model.fit(train)

#Here i'm not able to fit the model, as the x array of the diskdataset has a shape 3.
#So it raises a value error saying "Found array with dim 3. KernelRidge expected <= 2"
