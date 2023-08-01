from src import main
from src.models import multi_stage
from src.models.Hang2020 import spectral_network
from src.data import read_config
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

config = read_config("config.yml")
config["crop_dir"] = "/Users/benweinstein/Dropbox/Weecology/IFAS SEED/67ec871c49cf472c8e1ae70b185addb1"
m = multi_stage.MultiStage.load_from_checkpoint("/Users/benweinstein/Dropbox/Weecology/IFAS SEED/results/ac7b4194811c4bdd9291892bccc4e661.pt", config=config)
s = spectral_network(bands=config["bands"], classes=m.num_classes[4])
s.load_state_dict(m.models[4].model.year_models[2].state_dict())

dataset = m.level_2_train_ds
test_predictions = []
for batch in dataset:
    individual, inputs, label = batch
    prediction = s(inputs["HSI"][2].unsqueeze(0))[-1].detach()
    test_predictions.append(prediction)

test_predictions = np.vstack(test_predictions)

cmap = cm.get_cmap('tab20')
fig, ax = plt.subplots(figsize=(8,8))
num_categories = m.num_classes
for lab in range(num_categories):
    indices = test_predictions==lab
    ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=np.array(cmap(lab)).reshape(1,4), label = lab ,alpha=0.5)
ax.legend(fontsize='large', markerscale=2)
plt.show()