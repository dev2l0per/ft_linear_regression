import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from estimate_one import estimate, getTheta

def	getData():
	return pd.read_csv('./data.csv', dtype=np.float64).to_numpy()

def normalize(data):
	tmp_data = data.copy()
	tmp_data[:, 0] = (tmp_data[:, 0] - np.mean(tmp_data[:, 0])) / np.std(tmp_data[:, 0])
	tmp_data[:, 1] = (tmp_data[:, 1] - np.mean(tmp_data[:, 1])) / np.std(tmp_data[:, 1])
	return tmp_data

def denormalize(data_normal, data):
	data_normal[:, 0] = data_normal[:, 0] * np.std(data[:, 0]) + np.mean(data[:, 0])
	data_normal[:, 1] = data_normal[:, 1] * np.std(data[:, 1]) + np.mean(data[:, 1])
	return data_normal

def plot_data(data, x, y):
	plt.plot(data[:, 0], data[:, 1], 'o')
	plt.plot(x, y)
	plt.ylabel("Price")
	plt.xlabel("Km")
	plt.show()

def estimate_price(data, theta):
	data[:, 1] = theta[0] + theta[1] * data[:, 0]
	return data

def train(data, theta, learning_rate, iteration):
	for ith in range(iteration):
		temp_theta = theta[len(theta) - 1]
		weight_arr = []
		bias_arr = []
		for item in data:
			bias_arr.append(estimate(item[0], temp_theta) - item[1])
			weight_arr.append((estimate(item[0], temp_theta) - item[1]) * item[0])
		temp_theta[0] = temp_theta[0] - (learning_rate * np.mean(bias_arr))
		temp_theta[1] = temp_theta[1] - (learning_rate * np.mean(weight_arr))
		theta = np.concatenate((theta, temp_theta.reshape((1, -1))), axis = 0)
		print('theta0 : ' + str(temp_theta[0]) + ' theta1 : ' + str(temp_theta[1]))
	return theta

def	main():
	learning_rate = 0.5
	iteration = 1000

	theta = getTheta()
	data = getData()
	data_normalize = normalize(data)

	train_theta = train(data_normalize, theta, learning_rate, iteration)

	result = estimate_price(data_normalize, train_theta[len(train_theta) - 1])
	data_denormalize = denormalize(result, data)

	plot_data(data, data_denormalize[:, 0], data_denormalize[:, 1])

if __name__ == '__main__':
	main()