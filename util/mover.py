from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtGui import *
import math

class Mover(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setMinimumSize(1, 30)
        self.px = 0
        self.py = 0
        self.last_move = False
        self.rotation = 0
        
    def setRotation(self, angle):
        self.rotation = angle
        self.update()
        
    def paintEvent(self, e):
        qp = QtGui.QPainter()
        qp.begin(self)
        self.drawWidget(qp)
        qp.end()
        
    def drawWidget(self, qp):
        qp.translate(100,110)
        qp.rotate(self.rotation)
        
        font = QtGui.QFont('Serif', 11, QtGui.QFont.Light)
        qp.setFont(font)
        size = self.size()
        w = size.width()
        h = size.height()
        
        qp.setBrush(Qt.NoBrush)
        pen = QtGui.QPen()
        pen.setWidth(1)
        qp.setPen(pen)
        qp.setPen(QColor(255, 40, 40))
        for d in range(10, 80, 20):
            qp.drawEllipse(- d, -d, d*2, d*2)
            
        qp.setPen(QColor(0, 0, 0))
        qp.setBrush(Qt.SolidPattern)
        qp.drawEllipse(int(self.px - 5), int(self.py - 5), 10, 10)
        
        qp.setPen(QPen(Qt.black,1, Qt.SolidLine))
        qp.drawLine(-80,0,80,0)
        qp.drawLine(0,-80,0,80)
        
        # Draw cardinal directions with proper rotation
        # East
        qp.save()
        qp.translate(-95, 5)
        qp.rotate(-self.rotation)
        qp.drawText(0, 0, "E")
        qp.restore()
        
        # West
        qp.save()
        qp.translate(85, 5)
        qp.rotate(-self.rotation)
        qp.drawText(0, 0, "W")
        qp.restore()
        
        # North
        qp.save()
        qp.translate(-5, -87)
        qp.rotate(-self.rotation)
        qp.drawText(0, 0, "N")
        qp.restore()
        
        # South
        qp.save()
        qp.translate(-3, 97)
        qp.rotate(-self.rotation)
        qp.drawText(0, 0, "S")
        qp.restore()
        
    def mousePressEvent(self, event):
        dx = event.pos().x() - 100
        dy = event.pos().y() - 100
        angle = math.radians(self.rotation)
        self.px = dx * math.cos(angle) + dy * math.sin(angle)
        self.py = -dx * math.sin(angle) + dy * math.cos(angle)
        self.update()
        self.last_move = False
        
    def mouseMoveEvent(self, event):
        dx = event.pos().x() - 100
        dy = event.pos().y() - 100
        print(dx,dy)
        angle = math.radians(self.rotation)
        self.px = dx * math.cos(angle) + dy * math.sin(angle)
        self.py = -dx * math.sin(angle) + dy * math.cos(angle)
        self.update()
        
    def mouseReleaseEvent(self, event):
        self.px = 0
        self.py = 0
        self.last_move = True
        self.update()
        
    def moving(self):
        if (self.last_move):
            self.last_move = False
            return True
        return(self.px != 0 or self.py != 0)
        
    def rate(self):
        angle = math.radians(self.rotation)
        print(self.px, self.py)
        rx = self.px * math.cos(-angle) - self.py * math.sin(-angle)
        ry = self.px * math.sin(-angle) + self.py * math.cos(-angle)
        return rx/10.0, ry/10.0

class RotationSlider(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        # Initialize settings
        self.settings = QSettings('AstronomicalInstruments', 'Telescope')
        self.initUI()
        
    def initUI(self):
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Add rotation label
        self.label = QtWidgets.QLabel("Rotation:")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)
        
        # Add spin box with saved value
        self.spinBox = QtWidgets.QSpinBox()
        self.spinBox.setMinimum(0)
        self.spinBox.setMaximum(360)
        saved_rotation = self.settings.value('rotation', 0, type=int)
        self.spinBox.setValue(saved_rotation)
        self.spinBox.setSuffix("°")
        layout.addWidget(self.spinBox)
        
        # Add slider with saved value
        self.slider = QtWidgets.QSlider(Qt.Vertical)
        self.slider.setMinimum(0)
        self.slider.setMaximum(360)
        self.slider.setValue(saved_rotation)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksLeft)
        self.slider.setTickInterval(45)
        self.slider.setMinimumHeight(130)
        layout.addWidget(self.slider, 1)
        
        layout.addStretch()
        
        self.setLayout(layout)
        self.setMaximumWidth(100)
        
        # Connect signals with save functionality
        self.slider.valueChanged.connect(self.spinBox.setValue)
        self.slider.valueChanged.connect(self.save_rotation)
        self.spinBox.valueChanged.connect(self.slider.setValue)
        self.spinBox.valueChanged.connect(self.save_rotation)
        
    def save_rotation(self, value):
        """Save rotation value to settings"""
        self.settings.setValue('rotation', value)
        self.settings.sync()  # Ensure settings are written to disk

class CombinedWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create rotation slider
        self.rotationSlider = RotationSlider()
        layout.addWidget(self.rotationSlider)
        
        # Create mover
        self.mover = Mover()
        self.mover.setMinimumSize(200, 200)
        layout.addWidget(self.mover, 1)
        
        self.setLayout(layout)
        
        # Set initial rotation from saved value
        initial_rotation = self.rotationSlider.settings.value('rotation', 0, type=int)
        self.mover.setRotation(initial_rotation)
        
        # Connect rotation changes to mover
        self.rotationSlider.slider.valueChanged.connect(self.mover.setRotation)
        
    def getRotation(self):
        """Get current rotation value"""
        return self.rotationSlider.slider.value()
    
    def setRotation(self, value):
        """Set rotation value"""
        self.rotationSlider.slider.setValue(value)
        self.rotationSlider.save_rotation(value)

if __name__ == '__main__':
    import sys
    
    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QMainWindow()
    
    combined = CombinedWidget()
    window.setCentralWidget(combined)
    window.resize(400, 300)
    window.show()
    
    sys.exit(app.exec_())