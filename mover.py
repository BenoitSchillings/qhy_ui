from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *

class Mover(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        self.setMinimumSize(1, 30)
        self.px = 0
        self.py = 0
        self.last_move = False




    def paintEvent(self, e):

        qp = QtGui.QPainter()
        qp.begin(self)
        self.drawWidget(qp)
        qp.end()

    def drawWidget(self, qp):
        qp.translate(100,100)
        qp.rotate(0)
        
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
        qp.drawEllipse(self.px - 5, self.py -5, 10,10)



        qp.setPen(QPen(Qt.black,1, Qt.SolidLine))
        qp.drawLine(-80,0,80,0)
        qp.drawLine(0,-80,0,80)

        qp.drawText(-95, 5, "E")
        qp.drawText(85, 5, "W")

        qp.drawText(-5, 13-100, "N")
        qp.drawText(-3, 97, "S")


    def mousePressEvent(self, event):
        self.px = event.pos().x() - 100
        self.py = event.pos().y() - 100
        self.update()
        self.last_move = False


    def mouseMoveEvent(self, event):
        self.px = event.pos().x() - 100
        self.py = event.pos().y() - 100
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
        return self.px/10.0, self.py/10.0

