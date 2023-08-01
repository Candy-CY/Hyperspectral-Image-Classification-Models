import sys
sys.path.append("/home/b.weinstein/DeepTreeAttention")
from src import start_cluster
from src import multonomial
client = start_cluster.start(cpus=150)

for x in range(100):
    multonomial.wrapper(iteration=x, client=client, savedir="/blue/ewhite/b.weinstein/DeepTreeAttention/results/06ee8e987b014a4d9b6b824ad6d28d83")
