import sys
import pandas as pd
import numpy as np

def	estimate(inputMileage, theta):
	return theta[0] + (theta[1] * inputMileage)

def getTheta():
	try:
		theta = pd.read_csv("./theta.csv").to_numpy()
	except:
		theta = np.zeros((1, 2))
		df = pd.DataFrame(theta)
		df.to_csv('./theta.csv', index=False)
	return theta

def	main():
	theta = getTheta()
	print(theta)
	try:
		print('Please input your Mileage : ', end = "")
		inputMileage = float(input())
		estimatePrice = estimate(inputMileage, theta[0])
		if estimatePrice < 0:
			exit('No Price')
		print('Estimate Price : {}'.format(int(estimatePrice)))
	except EOFError:
		exit('Input EOF')
	except:
		exit('Command Error')

if __name__ == '__main__':
	main()