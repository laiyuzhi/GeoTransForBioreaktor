
import sys
sys.path.append('E:\\Program Files\\Abschlussarbeit\\GeoTransForBioreaktor-main\\geoTrans')

import re
import utils.Config as cfg
print(cfg.BATCH_SIZE)
string="Speed200"
print(type(int(re.findall(r"\d+",string)[0])))


label = [1, 200]
img = 'asd0//asdas'

print([img,label])
