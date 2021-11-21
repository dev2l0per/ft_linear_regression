import sys
import os
import pandas as pd
import numpy as np

def	estimate(inputMileage, theta,):
	return theta[0] + (theta[1] * inputMileage)

def normalize(input, data):
	input = (input - np.mean(data[:, 0])) / np.std(data[:, 0])
	return input

def denormalize(input, data):
	input = input * np.std(data[:, 1]) + np.mean(data[:, 1])
	return input

def getTheta():
	try:
		if (os.path.exists("./theta_after_train.csv")):
			theta = pd.read_csv("./theta_after_train.csv").to_numpy()
		else:
			theta = pd.read_csv("./theta.csv").to_numpy()
	except:
		theta = np.zeros((1, 2))
		df = pd.DataFrame(theta)
		df.to_csv('./theta.csv', index=False)
	return theta

def getData():
	try:
		data = pd.read_csv("./data.csv").to_numpy()
	except:
		exit('No Data CSV File')
	return data

def	main():
	theta = getTheta()
	data = getData()
	print(theta)
	try:
		print('Please input your Mileage : ', end = "")
		inputMileage = float(input())
		estimatePrice = estimate(normalize(inputMileage, data), theta[len(theta) - 1])
		print('Estimate Price : {}'.format(int(denormalize(estimatePrice, data))))
	except EOFError:
		exit('Input EOF')
	except:
		exit('Command Error')

if __name__ == '__main__':
	main()