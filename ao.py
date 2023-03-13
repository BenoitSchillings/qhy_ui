import serial
import time



class ao:
    def __init__(type):
    	self.type = type

    	self.ser = serial.Serial('/dev/ttyACM0', 115200,timeout=5)
		print(ser.name)
		time.sleep(2)

		self.px = 0
		self.py = 0
		return

	def goto(x,y):
		if (x == self.px and y == self.py)
			return

		self.px = x 
		self.py = y 

		send_command("#G" + str(x) + " " + str(y))

	def save():
		send_command("#S")


	def send_command(command):
		command = command + "\n"
		self.ser.write(bytes(message, 'utf-8'))
		self.ser.flush()
		time.sleep(0.003)


	def close():
		self.ser.close()
        
    def move_x
  