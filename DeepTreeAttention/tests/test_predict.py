import os
import pytest
from src import predict
import geopandas as gpd
from src.models import multi_stage, dead
from pytorch_lightning import Trainer
import pandas as pd
import cProfile
import pstats
from pytorch_lightning import Trainer

#Training module
@pytest.fixture()
def species_model_path(config, dm, ROOT, tmpdir):
    config["batch_size"] = 16    
    m  = multi_stage.MultiStage(train_df=dm.train, test_df=dm.test, crowns=dm.crowns, config=config)    
    m.ROOT = "{}/tests/".format(ROOT)
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(m)
    trainer.save_checkpoint("{}/model.pl".format(tmpdir))
    
    return "{}/model.pl".format(tmpdir)
    
def test_predict_tile(species_model_path, config, ROOT, tmpdir):
    rgb_path = "{}/tests/data/2019_D01_HARV_DP3_726000_4699000_image_crop_2019.tif".format(ROOT)
    config["HSI_sensor_pool"] = "{}/tests/data/hsi/*.tif".format(ROOT)
    config["CHM_pool"] = None
    config["prediction_crop_dir"] = tmpdir    
    
    dead_model = dead.AliveDead(config)
    trainer = Trainer(fast_dev_run=True)    
    trainer.fit(dead_model)
    dead_model_path = "{}/dead_model.pl".format(tmpdir)
    trainer.save_checkpoint(dead_model_path)
    
    crowns = predict.find_crowns(rgb_path, config, dead_model_path=dead_model_path)
    assert len(crowns.individual.unique()) == crowns.shape[0]
        
    crowns.to_file("{}/crowns.shp".format(tmpdir))
    
    crown_annotations_path = predict.generate_prediction_crops(crowns=crowns, config=config)
    crown_annotations = gpd.read_file(crown_annotations_path)
    
    # Assert that the geometry is correctly mantained
    assert crown_annotations.iloc[0].geometry.bounds == crowns[crowns.individual==crown_annotations.iloc[0].individual].iloc[0].geometry.bounds
    
    #There should be two of each individual, with the same geoemetry
    assert crown_annotations[crown_annotations.individual == crown_annotations.iloc[0].individual].shape[0] == 2
    assert len(crown_annotations[crown_annotations.individual == crown_annotations.iloc[0].individual].bounds.minx.unique()) == 1
    
    m = multi_stage.MultiStage.load_from_checkpoint(species_model_path, config=config)        
    trainer = Trainer(fast_dev_run=False, max_steps=1, limit_val_batches=1)
    
    trees = predict.predict_tile(
        crown_annotations=crown_annotations_path,
        m=m,
        trainer=trainer,
        filter_dead=True,
        savedir=tmpdir,
        config=config)
    
    assert all([x in trees.columns for x in ["geometry","ens_score","ensembleTaxonID"]])
    assert trees.iloc[0].geometry.bounds == trees[trees.individual==trees.iloc[0].individual].iloc[1].geometry.bounds
