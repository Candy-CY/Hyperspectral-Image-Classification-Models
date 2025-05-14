# -*- coding: utf-8 -*-
import os
import urllib


hsi_url = ['http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat',
           'http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat',
           'http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',
           'http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat',
           'http://www.ehu.es/ccwintco/uploads/2/26/KSC.mat',
           'http://www.ehu.es/ccwintco/uploads/a/a6/KSC_gt.mat']

for i in range(6):
    file_name = hsi_url[i].split('/')[-1]
    data_path = 'datasets/'+str(file_name)
    if os.path.exists(data_path) == False:
        print("Downloading data file from %s to %s" % (hsi_url[i], data_path))
        urllib.request.urlretrieve(url=hsi_url[i], filename=data_path)
        print(str(file_name)+" is Successfully downloaded")
    else:
        print(str(file_name) + " already exists")

print('All HSI dataset have already existed!')
