from orion_ao import *
import time


import sys

# Check if there are 2 arguments provided
if len(sys.argv) != 3:
    print("Error: Please provide 2 integer arguments.")
    sys.exit(1)

# Extract the arguments
try:
    number1 = int(sys.argv[1])
    number2 = int(sys.argv[2])
except ValueError:
    print("Error: Arguments must be integers.")
    sys.exit(1)


ao = ao()

ao.set_ao(number1, number2)

#for i in range(300):
#    ao.set_ao(i%10, i %11)
#    time.sleep(0.2)