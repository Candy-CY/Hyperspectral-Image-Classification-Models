#test megaplot
from src import megaplot

def test_read_files(config, ROOT):
    formatted_data = megaplot.read_files(directory="{}/tests/data/MegaPlots/".format(ROOT), config=config)
    assert all([x in formatted_data.columns for x in ["individual","plotID","siteID","taxonID"]])
