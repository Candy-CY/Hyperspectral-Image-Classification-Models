#crop random dataset
import glob
import sys
sys.path.append("/home/b.weinstein/DeepTreeAttention")

from src.data import read_config
import os
from src import neon_paths
from src.start_cluster import start
import rasterio
import random
import re
import numpy as np
from rasterio.windows import Window
from distributed import wait
import pandas as pd
import h5py
import json
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds

def crop(bounds, sensor_path, savedir = None, basename = None):
    """Given a 4 pointed bounding box, crop sensor data"""
    #dst_crs = 'EPSG:4326'
    
    left, bottom, right, top = bounds 
    src = rasterio.open(sensor_path)        
    img = src.read(window=rasterio.windows.from_bounds(left, bottom, left, top, transform=src.transform)) 
    res = src.res[0]
    
    height = (top - bottom)/res
    width = (right - left)/res   
    
    #transform, width, height = calculate_default_transform(
        #src.crs, dst_crs, width, height,*bounds)
    
    if savedir:
        profile = src.profile
        profile.update(
            height=height,
            width=width,
            transform=src.transform,
            crs=src.crs
        )
        filename = "{}/{}.tif".format(savedir, basename)
        with rasterio.open(filename, "w",**profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=src.transform,
                    dst_crs=src.crs,
                    resampling=Resampling.nearest)
            dst.write(img)
    if savedir:
        return filename
    else:
        return img 
    
def random_crop(config, iteration): 
    hsi_tif_pool = pd.read_csv("data/hsi_tif_pool.csv", index_col=0)["0"]
    rgb_pool = pd.read_csv("data/rgb_pool.csv", index_col=0)["0"]
    CHM_pool = pd.read_csv("data/CHM_pool.csv", index_col=0)["0"]
    hsi_pool = pd.read_csv("data/hsi_pool.csv", index_col=0)["0"]
    
    geo_index = re.search("(\d+_\d+)_image", os.path.basename(random.choice(hsi_tif_pool))).group(1)
    rgb_tiles = [x for x in rgb_pool if geo_index in x]
    if len(rgb_tiles) < 3:
        return None
    chm_tiles = [x for x in CHM_pool if geo_index in x]
    if len(chm_tiles) < 3:
        return None    
    if len([x for x in hsi_pool if geo_index in x]) < 3:
        return None
    
    #Get .tif from the .h5
    hsi_tifs = neon_paths.lookup_and_convert(rgb_pool=rgb_pool, hyperspectral_pool=hsi_pool, savedir=config["HSI_tif_dir"], geo_index=geo_index, all_years=True)           
    hsi_tifs = [x for x in hsi_tifs if not "neon-aop-products" in x]
    
    #HSI metadata
    hsi_h5 = [x for x in hsi_pool if geo_index in x]
  
    metadata_dicts = []
    for index, h5 in enumerate(hsi_h5):
        hdf5_file = h5py.File(h5, 'r')
        file_attrs_string = str(list(hdf5_file.items()))
        file_attrs_string_split = file_attrs_string.split("'")
        sitename = file_attrs_string_split[1]        
        metadata = {}
        metadata["siteID"] = sitename
        reflArray = hdf5_file[sitename]['Reflectance']        
        metadata_dicts.append(metadata)
    
    #year of each tile
    rgb_years = [neon_paths.year_from_tile(x) for x in rgb_tiles]
    hsi_years = [os.path.splitext(os.path.basename(x))[0].split("_")[-1] for x in hsi_tifs]
    chm_years = [neon_paths.year_from_tile(x) for x in chm_tiles]
    
    #Years in common
    selected_years = list(set(rgb_years) & set(hsi_years) & set(chm_years))
    selected_years = [x for x in selected_years if int(x) > 2017]
    selected_years.sort()
    selected_years = selected_years[-3:]
    if len(selected_years) < 3:
        print("not enough years")
        return None
    
    rgb_index = [index for index, value in enumerate(rgb_years) if value in selected_years]
    selected_rgb = [x for index, x in enumerate(rgb_tiles) if index in rgb_index]
    hsi_index = [index for index, value in enumerate(hsi_years) if value in selected_years]
    selected_hsi = [x for index, x in enumerate(hsi_tifs) if index in hsi_index]
    chm_index = [index for index, value in enumerate(chm_years) if value in selected_years]
    selected_chm = [x for index, x in enumerate(chm_tiles) if index in chm_index]
    
    #Ensure same order
    selected_rgb.sort()
    selected_chm.sort()
    selected_hsi.sort()
    
    if not all(np.array([len(selected_chm), len(selected_hsi), len(selected_rgb)]) == [3,3,3]):
        print("Not enough years")
        return None
    
    #Get window, mask out black areas
    src = rasterio.open(selected_rgb[0])   
    mask = src.read_masks(1)
    coordx = np.argwhere(mask==255)
    
    #Project bounds to web mercator
    #dst_crs = 'EPSG:4326'    
    
    #Get random coordinate away from edge, try 20 times
    counter=0
    while counter < 20:
        xsize, ysize = 640, 640
        random_index = random.randint(0, coordx.shape[0])
        xmin, ymin = coordx[random_index,:]
        window = Window(xmin, ymin, xsize, ysize)
        bounds = rasterio.windows.bounds(window, src.transform)
        if all([(bounds[2] - bounds[0] == 64), (bounds[3] - bounds[1])==64]):
            break
        else:
            counter = counter + 1
    
    #Project bounds to web mercator
    dst_crs = 'EPSG:4326'    
    orijbounds = transform_bounds(src.crs, dst_crs, *bounds)
    #orijbounds = bounds 
    projbounds = [abs(x) for x in orijbounds]
    center_x = np.mean([projbounds[0], projbounds[2]])
    center_x = str(center_x)
    center_x = center_x.replace(".","_")
    
    center_y = np.mean([projbounds[1], projbounds[3]])
    center_y = str(center_y)
    center_y = center_y.replace(".","_")
    center_coord = "{}N_{}W".format(center_y, center_x)
    
    coord_dir = "/blue/ewhite/b.weinstein/DeepTreeAttention/selfsupervised/{}".format(center_coord)
    try:
        os.mkdir(coord_dir)
    except:
        pass
    
    #crop rgb
    for tile in selected_rgb:
        year = "{}-01-01".format(os.path.basename(tile).split("_")[0])
        year_dir = os.path.join(coord_dir, year)
        try:
            os.mkdir(year_dir)
        except:
            pass
        
        filename = crop(
            bounds=bounds,
            sensor_path=tile,
            savedir=year_dir,
            basename="RGB")
    
    #Crop CHM
    for index, tile in enumerate(selected_chm):
        yr = "{}-01-01".format(selected_years[index])  
        year_dir = os.path.join(coord_dir,yr)        
        filename = crop(
            bounds=bounds,
            sensor_path=tile,
            savedir=year_dir,
            basename="CHM")
        
        selected_dict = metadata_dicts[index]
        selected_dict["bounds"] = orijbounds   
        selected_dict["epsg"] = str(src.crs)
        
        with open(os.path.join(year_dir,"metadata.json"), 'w') as convert_file:
            convert_file.write(json.dumps(selected_dict, indent=4, sort_keys=True))
    
    #HSI
    for index, tile in enumerate(selected_hsi):
        yr = "{}-01-01".format(selected_years[index])        
        year_dir = os.path.join(coord_dir, yr)
        filename = crop(
            bounds=bounds,
            sensor_path=tile,
            savedir=year_dir,
            basename="HSI")

if __name__ == "__main__":
    client = start(cpus=80, mem_size = "25GB")    
    config = read_config("config.yml")    
    rgb_pool = glob.glob("/orange/ewhite/NeonData/*/DP3.30010.001/**/Camera/**/*.tif", recursive=True)
    rgb_pool = [x for x in rgb_pool if not "classified" in x]
    pd.Series(rgb_pool).to_csv("data/rgb_pool.csv")
    
    hsi_pool = glob.glob("/orange/ewhite/NeonData/*/DP3.30006.001/**/Reflectance/*.h5", recursive=True)
    hsi_pool = [x for x in hsi_pool if not "neon-aop-products" in x]
    pd.Series(hsi_pool).to_csv("data/hsi_pool.csv")
    
    CHM_pool = glob.glob("/orange/ewhite/NeonData/**/CanopyHeightModelGtif/*.tif", recursive=True)
    pd.Series(CHM_pool).to_csv("data/CHM_pool.csv")
    
    hsi_tif_pool = glob.glob(config["HSI_tif_dir"]+"*")
    pd.Series(hsi_tif_pool).to_csv("data/hsi_tif_pool.csv")
    
    futures = []
    
    for x in range(100000):
        future = client.submit(random_crop, 
                               config=config, 
                               iteration=x)
        futures.append(future)
    
    wait(futures)
    
    for x in futures:
        try:
            x.result()
        except Exception as e:
            print(e)
            
    # post process cleanup
    files = glob.glob("/blue/ewhite/b.weinstein/DeepTreeAttention/selfsupervised/**/*.tif",recursive=True)
    counts = pd.DataFrame({"basename":[os.path.basename(x) for x in files],"path":files}) 
    counts["geo_index"] = counts.path.apply(lambda x: os.path.dirname(os.path.dirname(x)))
    less_than_3 = counts.groupby("geo_index").basename.value_counts().reset_index(name="geo")
    to_remove = less_than_3[less_than_3.geo < 3].geo_index
    for x in counts[counts.basename.isin(to_remove)].path:
        os.remove(x)
