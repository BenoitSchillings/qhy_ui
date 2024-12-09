from fli_focuser import *



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


foc = focuser()

print(foc.status())
foc.move_focus(number1)
print(foc.get_pos())