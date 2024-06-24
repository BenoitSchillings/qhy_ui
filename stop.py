import skyx
import time
import argparse
import random



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	args = parser.parse_args()


	sky = skyx.sky6RASCOMTele()
	sky.Connect()

	sky.stop()
