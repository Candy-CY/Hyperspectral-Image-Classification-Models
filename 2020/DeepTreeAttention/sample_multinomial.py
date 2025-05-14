from src.multinomial import *
from src import start_cluster
import glob
import pandas as pd

client = start_cluster.start(cpus=50, mem_size="10GB")
for x in range(100):
    wrapper(client=client, iteration=x, experiment_key="06ee8e987b014a4d9b6b824ad6d28d83")
