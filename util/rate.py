import skyx
import time
import argparse
import random



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-ra", type=float, default = 1.0, help="ra rate")
	parser.add_argument("-dec", type=float, default = 1.0, help="dec rate")
	args = parser.parse_args()
	print(args)

	sky = skyx.sky6RASCOMTele()
	sky.Connect()

	sky.rate(args.ra, args.dec)
