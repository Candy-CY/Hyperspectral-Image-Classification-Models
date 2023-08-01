#Test data module
from src import data
import pandas as pd
import copy

def test_TreeData_setup(config, ROOT):
    #One site's worth of data
    local_config = copy.deepcopy(config)
    local_config["use_data_commit"] = None
    
    csv_file = "{}/tests/data/sample_neon.csv".format(ROOT)               
    dm = data.TreeData(config=local_config, csv_file=csv_file, data_dir="{}/tests/data".format(ROOT), debug=True) 
    
    test = pd.read_csv("{}/tests/data/processed/test.csv".format(ROOT))
    train = pd.read_csv("{}/tests/data/processed/train.csv".format(ROOT))
    
    assert not test.empty
    assert not train.empty
    assert not any([x in train.image_path.unique() for x in test.image_path.unique()])
    assert all([x in ["image_path","label","site","taxonID","siteID","plotID","individualID","point_id","box_id","RGB_tile","tile_year"] for x in train.columns])
    assert len(train.tile_year.unique()) > 1
    
def test_TreeDataset(m, dm, config):
    #Train loader
    ds = data.TreeDataset(dm.train, config=dm.config)
    individuals, inputs, label = ds[0]
    
    assert len(inputs["HSI"]) == len(dm.train.tile_year.unique())
    assert inputs["HSI"][0].shape == (config["bands"], config["image_size"], config["image_size"])
    
def test_sample_plots(dm, config):
    train, test = data.sample_plots(shp=dm.crowns, min_test_samples=10, min_train_samples=10)
    assert not train.empty
    assert train[train.individual.isin(test.individual)].empty
