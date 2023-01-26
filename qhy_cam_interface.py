import qhyccd

class qhy_cam:
    def __init__(self, temp, exp, gain, crop, cam_name):
        self.qc = qhyccd.qhyccd(cam_name)
        self.dt = exp
        self.gain = gain
        self.qc.GetSize()
        self.qc.SetBit(16)
        self.qc.SetUSB(3)
       

        self.qc.SetOffset(144) #for guider
        #self.qc.SetOffset(100) #for imager

        self.qc.SetTemperature(temp)
        self.sizex = int(self.qc.image_size_x * crop)
        self.sizey = int(self.qc.image_size_y * crop)

        ddx = self.qc.image_size_x - self.sizex
        ddy = self.qc.image_size_y  -self.sizey
        ddx = ddx // 2
        ddy = ddy // 2

        self.qc.SetROI(ddx,ddy,ddx + self.sizex,ddy + self.sizey)
        self.qc.SetExposure(self.dt*1000)
       
        self.qc.SetGain(gain)
        
   
    def get_frame(self):        
        self.frame = self.qc.GetSingleFrame()
        return self.frame
        
    def start(self):
        self.running = 1
        self.qc.Begin()
        
    def close(self):
        self.running = 0
        #self.qc.StopLive()

    def size_x(self):
        return self.sizex
    
    def size_y(self):
        return self.sizey

    def name(self):
        return self.qc.GetName()


