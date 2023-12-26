from orion_ao import *



import sys

# Check if there are 1 arguments provided
if len(sys.argv) != 2:
    print("Error: Please provide 1 integer arguments.")
    sys.exit(1)

# Extract the arguments
try:
    number1 = int(sys.argv[1])
except ValueError:
    print("Error: Arguments must be integers.")
    sys.exit(1)


ao = orion_ao()

ao.rotate_to_angle(number1)
