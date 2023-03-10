import serial
import time
import logging as log

log.basicConfig(level=log.INFO)



class ao:
    def __init__(self, type=""):
    	self.type = type

    	self.ser = serial.Serial('/dev/ttyACM0', 115200,timeout=5)
		print(ser.name)
		time.sleep(2)
		self.send_command("#G0 0")
		self.px = 0
		self.py = 0
		return

	def goto(self, x,y):
		if (x == self.px and y == self.py):
			return

		self.px = x 
		self.py = y 

		self.send_command("#G" + str(x) + " " + str(y))


	def zero(self):
		self.goto(0,0)
		time.sleep(0.3)


	def move(self, x,y):
		if (x == 0 and y == 0):
			return

		self.px = self.px + x
		self.py = self.py + y

		self.send_command("#G" + str(self.px) + " " + str(self.py))


	def save(self):
		self.send_command("#S")


	def send_command(self, command):
		log.info("ao %s", command)
		command = command + "\n"
		self.ser.write(bytes(message, 'utf-8'))
		self.ser.flush()
		time.sleep(0.003)


	def close(self):
		self.goto(0,0)
		time.sleep(0.1)
		self.save()

		self.ser.close()
        
    def __del__(self):
    	self.close():
  