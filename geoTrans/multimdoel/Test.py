
import sys
sys.path.append('/mnt/projects_sdc/lai/GeoTransForBioreaktor/geoTrans')

import re
import utils.Config as cfg
print(cfg.BATCH_SIZE)
string = "Speed200Luft2"
print([int(x) for x in re.findall(r"\d+", string)])


label = [1, 200]
img = 'asd0//asdas'

print([img,label])
