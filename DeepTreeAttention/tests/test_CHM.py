#Test CHM height rules
import pandas as pd
from src import CHM

def test_height_rules():   
    df = pd.DataFrame({"CHM_height":[11,20,5, 0.5, 10, None],"height":[6, 19, 7, 5, None, 10]})
    df = CHM.height_rules(df, min_CHM_height=1, max_CHM_diff=4, CHM_height_limit=8)
    assert df.shape[0] == 3
    