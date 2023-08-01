#Plot abundance distribution
from glob import glob
import os
import pandas as pd
import geopandas as gpd
from src import start_cluster

client = start_cluster.start(cpus=100,mem_size="5GB")

#Same data

#species_model_paths = #"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/06ee8e987b014a4d9b6b824ad6d28d83.pt",
                       #"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/ac7b4194811c4bdd9291892bccc4e661.pt",
species_model_paths = ["/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/b629e5365a104320bcec03843e9dd6fd.pt",
                       "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/5ac9afabe3f6402a9c312ba4cee5160a.pt",
                       "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/46aff76fe2974b72a5d001c555d7c03a.pt",
                       "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/63bdab99d6874f038212ac301439e9cc.pt",
                       "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/c871ed25dc1c4a3e97cf3b723cf88bb6.pt",
                       "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/6d45510824d6442c987b500a156b77d6.pt",
                       "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/83f6ede4f90b44ebac6c1ac271ea0939.pt",
                       "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/47ee5858b1104214be178389c13bd025.pt",
                       "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/1ccdc11bdb9a4ae897377e3e97ce88b9.pt",
                        "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/3c7b7fe01eaa4d1b8a1187b792b8de40.pt",
                        "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/3b6d9f2367584b3691de2c2beec47beb.pt",
                        "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/509ef67c6050471e83199d2e9f4f3f6a.pt",
                        "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/ae7abdd50de04bc9970295920f0b9603.pt",
                        "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/d2180f54487b45269c1d86398d7f0fb8.pt",
                        "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/6f9730cbe9ba4541816f32f297b536cd.pt",
                        "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/71f8ba53af2b46049906554457cd5429.pt",
                        "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/6a28224a2dba4e4eb7f528d19444ec4e.pt",
                        "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/b9c0111b1dc0420b84e3b6b79da4e166.pt"]                       

def read_shp(path):
    gdf = gpd.read_file(path)
    gdf = gdf.groupby("individual").apply(lambda x: x.head(1))    
    #limit by OSBS polygon
    boundary = gpd.read_file("/home/b.weinstein/DeepTreeAttention/data/raw/OSBSBoundary/OSBS_boundary.shp")
    boundary = boundary.to_crs("epsg:32617")
    intersects = gpd.clip(gdf, boundary)
    
    return intersects

futures = []
for species_model_path in species_model_paths:
    print(species_model_path)
    basename = os.path.splitext(os.path.basename(species_model_path))[0]
    input_dir = "/blue/ewhite/b.weinstein/DeepTreeAttention/results/{}/*_image.shp".format(basename)
    files = glob(input_dir)
    print(files)
    if len(files) == 0:
        continue
    counts = []
    futures = client.map(read_shp,files)
    shps = [x.result() for x in futures]
    combined_shps = pd.concat(shps)
    gpd_boundary = gpd.GeoDataFrame(combined_shps, geometry="geometry")
    gpd_boundary = gpd_boundary.reset_index(drop=True)
    gpd_boundary.to_file("/blue/ewhite/b.weinstein/DeepTreeAttention/results/{}/predictions.shp".format(basename))
