#test patches
import os
from src import patches
from src import __file__ as ROOT
import geopandas as gpd
import rasterio

ROOT = os.path.dirname(os.path.dirname(ROOT))   
    
def test_crop(tmpdir):
    gdf = gpd.read_file("{}/tests/data/crown.shp".format(ROOT))
    rgb_path = "{}/tests/data/2019_D01_HARV_DP3_726000_4699000_image_crop_2018.tif".format(ROOT)   
    patch = patches.crop(bounds=gdf.geometry[0].bounds,sensor_path=rgb_path, savedir=tmpdir, basename="test")
    img = rasterio.open(patch).read()
    assert img.shape[0] == 3    