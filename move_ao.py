from orion_ao import *



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


ao = orion_ao()

ao.set_ao(number1, number2)