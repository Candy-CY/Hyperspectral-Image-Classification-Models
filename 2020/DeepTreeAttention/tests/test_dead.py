#test dead
from src.models import dead
from src.predict import predict_crowns
from pytorch_lightning import Trainer
import torch
import numpy as np

def test_fit(config):
    trainer = Trainer(fast_dev_run=True)
    m = dead.AliveDead(config=config)
    trainer.fit(m)
    assert True

def test_validate(config):
    trainer = Trainer(fast_dev_run=True)
    m = dead.AliveDead(config=config)
    trainer.validate(m)
    assert True   
    
def test_predict(ROOT, config):
    trainer = Trainer(fast_dev_run=True)
    config["dead"]["batch_size"] = 2
    m = dead.AliveDead(config=config)
    rgb_path = "{}/tests/data/2019_D01_HARV_DP3_726000_4699000_image_crop_2018.tif".format(ROOT)   
    crowns = predict_crowns(rgb_path)
    ds = dead.utm_dataset(crowns=crowns, config=config)
    predictions = trainer.predict(m, m.predict_dataloader(ds))
    predictions = np.concatenate(predictions)
    assert predictions.shape[1] == 2
    
    #Produces non-unique predictions
    assert not len(np.unique(predictions[:,1])) == 1
