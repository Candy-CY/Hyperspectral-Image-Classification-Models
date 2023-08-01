#test metrics
from src import metrics
import pandas as pd

def test_site_confusion():
    y_true = [0,0,1,1,1]
    y_pred = [0,0,1,1,0]
    site_lists = {0:[0], 1:[0]}
    assert metrics.site_confusion(y_true, y_pred, site_lists) == 1
    
    y_true = [0,0,1,1,1]
    y_pred = [0,0,1,1,0]
    site_lists = {0:[1], 1:[0]}
    assert metrics.site_confusion(y_true, y_pred, site_lists) == 0    
    
def test_genus_confusion():
    y_true = [0,0,1,1,1]
    y_pred = [0,0,1,1,2]   
    
    scientific_dict = {0:"ACRU",1:"QUMU",2:"QUMLA"}
    
    #all error is within genus
    assert metrics.genus_confusion(y_true, y_pred, scientific_dict) == 1
    
    y_true = [0,0,1,1,1]
    y_pred = [0,0,1,1,0]   
    
    scientific_dict = {0:"ACRU",1:"QUMU",2:"QUMLA"}
    
    #all error is outside genus
    assert metrics.genus_confusion(y_true, y_pred, scientific_dict) == 0  