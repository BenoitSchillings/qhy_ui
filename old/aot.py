from ticlib import TicUSB
from time import sleep
import math

tic_a = TicUSB(serial_number='00438821')
tic_c = TicUSB(serial_number='00438845')

tic_a.set_max_speed(17000000)
tic_c.set_max_speed(17000000)

k = 3
tic_a.set_starting_speed(k*1500000)
tic_c.set_starting_speed(k*1500000)

mode = 1
tic_a.set_step_mode(mode)
tic_c.set_step_mode(mode)


def set_pos(motor, value):
    value = value * 1
    if (motor == 0):
        tic_a.set_target_position(value)

    if (motor == 2):
        tic_c.set_target_position(-value)

limit = 5

def restart():
    tic_a.energize()

    tic_c.energize()
    tic_a.exit_safe_start()
    tic_c.exit_safe_start()
    tic_a.set_decay_mode(1)
    tic_c.set_decay_mode(1)
    tic_a.set_current_limit(limit)
    tic_c.set_current_limit(limit)


tic_a.halt_and_set_position(0)
tic_c.halt_and_set_position(0)

restart()

for i in range(7):
    set_pos(0, -i*300)
    set_pos(1, -i*300)
    
    sleep(1)
    print(tic_a.get_current_position())

tic_a.halt_and_set_position(0)
tic_c.halt_and_set_position(0)

sleep(2)
set_pos(0, 450)
set_pos(1, 450)
#restart()

sleep(1)
#restart()

def set_xy(x, y):
    set_pos(0, 320+int(x))
    set_pos(2, 320+int(y))
    print(x,y)



while(True):
    for x in range(200000):
        #vx = 320.0*math.sin(x/14.0)
        #vy = 320.0*math.cos(x/14.0)
        #vx = 0
        #print(vx, vy)
        #set_xy(vx, vy)
        if (x % 2 == 0):
            set_xy(0, -300)
        else:
            set_xy(0, 300)

        sleep(0.9)



tic_a.deenergize()
#tic_b.deenergize()
tic_c.deenergize()
#tic_d.deenergize()
tic_a.enter_safe_start()