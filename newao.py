from src.ticlib import TicUSB
from time import sleep
import math

tic_a = TicUSB(serial_number='00438821')
tic_b = TicUSB(serial_number='00440812')
tic_c = TicUSB(serial_number='00438845')
tic_d = TicUSB(serial_number='00282643')

tic_a.set_max_speed(7000000)
tic_b.set_max_speed(7000000)
tic_c.set_max_speed(7000000)
tic_d.set_max_speed(7000000)

tic_a.set_starting_speed(2000000)
tic_b.set_starting_speed(2000000)
tic_c.set_starting_speed(2000000)
tic_d.set_starting_speed(2000000)

tic_a.set_step_mode(2)
tic_b.set_step_mode(2)
tic_c.set_step_mode(2)
tic_d.set_step_mode(2)

def set_pos(motor, value):
    if (motor == 0):
        tic_a.set_target_position(-value)
    if (motor == 1):
        tic_b.set_target_position(value)
    if (motor == 2):
        tic_c.set_target_position(value)
    if (motor == 3):
        tic_d.set_target_position(-value)


def restart():
    tic_a.energize()
    tic_b.energize()
    tic_c.energize()
    tic_d.energize()
    tic_a.exit_safe_start()
    tic_b.exit_safe_start()
    tic_c.exit_safe_start()
    tic_d.exit_safe_start()
    tic_a.set_decay_mode(1)
    tic_b.set_decay_mode(1)
    tic_c.set_decay_mode(1)
    tic_d.set_decay_mode(1)
    tic_a.set_current_limit(4)
    tic_b.set_current_limit(4)
    tic_c.set_current_limit(4)
    tic_d.set_current_limit(4)


tic_a.halt_and_set_position(0)
tic_b.halt_and_set_position(0)
tic_c.halt_and_set_position(0)
tic_d.halt_and_set_position(0)

restart()

for i in range(3):
    set_pos(0, -i*300)
    set_pos(1, -i*300)
    set_pos(2, -i*300)
    set_pos(3, -i*300)
    
    sleep(1)
    print(tic_a.get_current_position())

tic_a.halt_and_set_position(0)
tic_b.halt_and_set_position(0)
tic_c.halt_and_set_position(0)
tic_d.halt_and_set_position(0)

sleep(2)
set_pos(0, 170)
set_pos(1, 170)
set_pos(2, 170)
set_pos(3, 170)
#restart()

#tic_a.halt_and_set_position(100)
#tic_b.halt_and_set_position(100)
#tic_c.halt_and_set_position(100)
#tic_d.halt_and_set_position(100)

sleep(1)
#restart()

while(True):
    for x in range(200000):
        vx = math.sin(x/42.0)
        vy = math.cos(x/42.0)
        set_pos(0, 400+int(vx*160))
        set_pos(3, 400+-int(vx*160))
        set_pos(1, 400+int(vy*160))
        set_pos(2, 400+-int(vy*160))
        sleep(0.1)



tic_a.deenergize()
tic_b.deenergize()
tic_c.deenergize()
tic_d.deenergize()
tic_a.enter_safe_start()
