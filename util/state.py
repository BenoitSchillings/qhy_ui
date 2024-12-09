from util import *

import logging as log

log.basicConfig(level=log.INFO)


server = IPC(type = zmq.REP)
variables = {'': 0}
variables['bump'] = [0,0]

while(True):
	data = server.get()
	#print("data")

	if (data != None):
		#print(data)

		if (data[1] != -1):
			variables[data[0]] = data[1]

		server.send(variables[data[0]])


server.close()
