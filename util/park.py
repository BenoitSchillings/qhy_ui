import skyx
import time
import argparse
import random



if __name__ == "__main__":

	sky = skyx.sky6RASCOMTele()
	sky.Connect()

	sky.park()
