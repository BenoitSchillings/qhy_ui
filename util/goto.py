#goto.py

import zmq
import logging as log
from util import *
import argparse
import skyx

log.basicConfig(level=log.INFO)



sky = skyx.sky6RASCOMTele()

ipc = IPC()


parser = argparse.ArgumentParser()

args = parser.parse_args()

try:
    sky.Connect()
except:
    sky = None


from astroquery.vizier import Vizier
from astropy.io.votable import from_table, writeto, parse

Vizier.ROW_LIMIT = -1

catalog_list = Vizier.get_catalogs('VII/118/ngc2000')
cat = catalog_list[0]

import pickle

pickle.dump(cat, open('ngc.cat', 'wb'))
ff = pickle.load(open('ngc.cat', 'rb'))

#print(ff)

idx = np.where(cat['Name'] == '100')
ngc100_rows = cat[idx]

print(ngc100_rows)
if (sky == None):
	print("Please start theSKY TCP Server")
else:
	print("done")