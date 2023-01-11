
# import sys
# sys.path.append('/mnt/projects_sdc/lai/GeoTransForBioreaktor/geoTrans')

# import re
# import utils.Config as cfg
# print(cfg.BATCH_SIZE)
# string = "Speed200Luft2"
# print([int(x) for x in re.findall(r"\d+", string)])


# label = [1, 200]
# img = 'asd0//asdas'

# print([img,label])

from visdom import Visdom
import numpy as np
import time

wind = Visdom()
wind.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))

for step in range(10):
    loss = np.random.randn()
    wind.line([loss], [step], win='train_loss', update='append')
    time.sleep(0.5)
