import glob
import geopandas as gpd
import numpy as np
import os
import pandas as pd
import rasterio as rio
from src import patches
from src import neon_paths
from src.data import read_config
from src.start_cluster import start
from distributed import wait

config = read_config("config.yml")
shapefiles = glob.glob("/orange/idtrees-collab/draped/*.shp")
shapefiles = [x for x in shapefiles if "OSBS" in x]
np.random.shuffle(shapefiles)
rgb_pool = glob.glob(config["rgb_sensor_pool"], recursive=True)
HSI_pool = glob.glob(config["HSI_sensor_pool"], recursive=True)

client = start(cpus=50)

futures = []
for i in shapefiles:
    shp = gpd.read_file(i)
    basename = os.path.splitext(os.path.basename(i))[0]
    #get 100 random trees
    try:
        shp = shp.sample(n=1000)
    except:
        continue
    hsi_path = neon_paths.lookup_and_convert(bounds=shp.total_bounds, rgb_pool=rgb_pool, hyperspectral_pool=HSI_pool, savedir=config["HSI_tif_dir"])    
    for index, row in shp.iterrows():
        future = client.submit(patches.crop, bounds=row["geometry"].bounds, sensor_path=hsi_path, savedir="/orange/idtrees-collab/mining/", basename="{}_{}".format(basename, index))
        futures.append(future)

wait(futures)

def remove(x):
    i = rio.open(x).read()
    if not np.isfinite(i).all():
        os.remove(x)

#Make sure all data is valid.
images = glob.glob("/orange/idtrees-collab/mining/*.tif")
futures = client.map(remove, images)
wait(futures)


images = glob.glob("/orange/idtrees-collab/mining/*.tif")
mining = pd.DataFrame({"image_path":images})
mining.to_csv("/orange/idtrees-collab/mining/mining.csv")
