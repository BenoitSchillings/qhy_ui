import serial
import time
import logging as log

log.basicConfig(level=log.INFO)



class ao:
    def __init__(self, type=""):
        self.type = type
        print("open")
        self.px = 0
        self.py = 0
        self.ser = serial.Serial('/dev/ttyACM0', 115200,timeout=0.1)
        print(self.ser.name)
        time.sleep(2)

        self.send_command("#g0 0")

        return

    def goto(self, x,y):
        if (x == self.px and y == self.py):
            return

        self.px = x 
        self.py = y 
        
        self.send_command("#g" + str(x) + " " + str(y))




    def zero(self):
        self.goto(0,0)
        time.sleep(0.3)


    def move(self, x,y):
        if (x == 0 and y == 0):
            return

        self.px = self.px + x
        self.py = self.py + y

        self.send_command("#g" + str(self.px) + " " + str(self.py))


    def save(self):
        self.send_command("#s")

    def write_s(self, cmd):
    	#print(cmd)
    	self.ser.write(cmd)

    def send_command(self, command):
        log.info("ao %s", command)
        command = command + "\n"

        self.write_s(bytes(command, encoding = 'ascii'))
        self.ser.flush()
        time.sleep(0.003)


    def close(self):
        self.goto(0,0)
        time.sleep(0.1)
        self.save()

        self.ser.close()
        

if __name__ == "__main__":
	test_ao = ao()
	test_ao.goto(200,200)
	time.sleep(1)
	test_ao.goto(0,0)
	test_ao.close()
