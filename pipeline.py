import os
from random import random, randint
from mlflow import log_metric, log_param, log_artifacts

# First Step: 
# https://mlflow.org/docs/latest/tutorials-and-examples/tutorial.html
# Do this tutorial : You can copy and paste the code, but the point is you need to understand what
# every single line of code is doing.
# 
# Use the debugger, or print() statements, so you can watch exectution flow as it happens
# practice explaining it to youself and then when you're done, let's have a google-meeting
# and I will ask you heaps of questions about the code
# 

# Then after that, we can come up with a set of steps to do the QM7 Dataset!

if __name__ == "__main__":
    # Log a parameter (key-value pair)
    log_param("param1", randint(0, 100))

    # Log a metric; metrics can be updated throughout the run
    log_metric("foo", random())
    log_metric("foo", random() + 1)
    log_metric("foo", random() + 2)

    # Log an artifact (output file)
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    with open("outputs/test.txt", "w") as f:
        f.write("test complete")
    log_artifacts("outputs")
