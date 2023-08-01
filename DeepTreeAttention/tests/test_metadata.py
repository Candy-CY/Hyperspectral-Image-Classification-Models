#Test metadata model
from src.models import metadata
from src import data
from src import utils
import torch
import os
from pytorch_lightning import Trainer

ROOT = os.path.dirname(os.path.dirname(data.__file__)) 

def test_metadata():
    m = metadata.metadata(sites = 1, classes=10)
    sites = torch.zeros(20)     
    output = m(sites.int())
    assert output.shape == (20,10)
    
def test_metadata_sensor_fusion():
    sites = torch.zeros(20)
    image = torch.randn(20, 3, 11, 11)    
    
    m = metadata.metadata_sensor_fusion(bands=3, sites=1, classes=10)
    prediction = m(image, sites.int())
    assert prediction.shape == (20,10)

#def test_MetadataModel(config, dm):
    #model = metadata.metadata_sensor_fusion(sites=1, classes=3, bands=3)
    #m = metadata.MetadataModel(model=model, classes=3, label_dict=dm.species_label_dict, config=config)
    #trainer = Trainer(fast_dev_run=True)
    #trainer.fit(m,datamodule=dm)    
