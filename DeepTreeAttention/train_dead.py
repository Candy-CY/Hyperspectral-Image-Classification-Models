# Train Dead 
import comet_ml
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger
from src.models import dead
from src.data import read_config
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay

config = read_config("config.yml")
comet_logger = CometLogger(
    project_name="DeepTreeAttention",
    workspace=config["comet_workspace"],
    auto_output_logging="simple"
)    
comet_logger.experiment.add_tag("Dead")

trainer = Trainer(max_epochs=config["dead"]["epochs"], checkpoint_callback=False, gpus=config["gpus"], logger=comet_logger)
m = dead.AliveDead(config=config)

trainer.fit(m)
trainer.validate(m)
trainer.save_checkpoint("{}/{}.pl".format(config["dead"]["savedir"],comet_logger.experiment.id))

true_class = [x[1] for x in m.val_ds]
m.eval()
with torch.no_grad():
    predictions = [m(x[0].unsqueeze(0)) for x in m.val_ds]
predicted_class = [np.argmax(x.numpy()) for x in predictions]
predicted_scores = [np.max(x.numpy()) for x in predictions]

comet_logger.experiment.log_confusion_matrix(
    true_class,
    predicted_class,
    labels=["Alive","Dead"], index_to_example_function=dead.index_to_example, test_dataset=m.val_ds)    

precision, recall, thresholds = precision_recall_curve(y_true=true_class, probas_pred=predicted_scores)
disp = PrecisionRecallDisplay(precision=precision, recall=recall)
disp.plot()
comet_logger.experiment.log_figure("precision_recall")
