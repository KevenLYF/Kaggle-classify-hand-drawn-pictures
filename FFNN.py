from utility import cleanNoise, cleanNoise3, TrimImage, AugmentImages
from math import exp
import random
import pandas as pd
import numpy as np
import imp
import sklearn.metrics as skMetrics

def sigmoid(aX):
	return 1.0 / (1.0 + exp(-aX))

# Derivative of the sigmoid function
def sigmoidDerivative(output):
	return output * (1.0 - output)

# Initialize create layer with random weights
def createLayer(numIn, numOut):
    return [{'w':[random.random() for i in range(numIn + 1)]} for i in range(numOut)]

# Network Constructor
def constructNet(numInputs, numLayers, numOutputs):
    
    net = list()

    # Create all hidden layers
    hiddenLayer = createLayer(numInputs, numLayers)

    # Create output layer
    outputLayer = createLayer(numLayers, numOutputs)
    
    # Add them to the network
    net.append(hiddenLayer)
    net.append(outputLayer)
    
    return net

def feedForward(net, row):
	inputs = row
	for l in net:
		newIn = []
		for n in l:
			aX = preActivation(n['w'], inputs)
			n['out'] = sigmoid(aX)
			newIn.append(n['out'])
		inputs = newIn
	return inputs

def backpropagationStep(net, expected, i):
    layer = net[i]
    errors = list()
    if i == len(net)-1:
        for j in range(len(layer)):
            n = layer[j]
            errors.append(expected[j] - n['out'])
    else:
        for j in range(len(layer)):
            error = 0.0
            for neuron in net[i + 1]:
                error += (neuron['w'][j] * neuron['d'])
            errors.append(error)
    for j in range(len(layer)):
        n = layer[j]
        n['d'] = errors[j] * sigmoidDerivative(n['out'])

# Backpropagate
def backpropagation(net, expected):
    for i in reversed(range(len(net))):
        backpropagationStep(net, expected, i)

def weightUpdate(network, input, learningRate):
	for i in range(len(network)):
		inputs = input[:-1]
		if i != 0:
			inputs = [n['out'] for n in network[i - 1]]
		for n in network[i]:
			for j in range(len(inputs)):
				n['w'][j] += learningRate * n['d'] * inputs[j]
			n['w'][-1] += learningRate * n['d']

def preActivation(w, x):
    a = 0
    for i in range(len(w)-1):
        a += w[i] * x[i]
    b = w[-1]
    a += b
    return a

def trainingStep(epoch, net, numOutput, learningRate, trainingSet):
    error = 0
    for input in trainingSet:
        out = feedForward(net, input)
        y = [0 for i in range(numOutput)]
        y[int(input[-1])] = 1
        error += 0.5*sum([(y[i]-out[i])**2 for i in range(len(y))])
        backpropagation(net, y)
        weightUpdate(net, input, learningRate)
    
    return error

def train(net, trainingSet, learningRate, epochs, numOutput):
    lastError = 0
    for epoch in range(epochs-1):
        error = trainingStep(epoch, net, numOutput, learningRate, trainingSet)

        # Check if we are not learning anymore, stop if it is the case
        if error - lastError == 0:
            break
        lastError = error

def getData():
    IMG_SIZE = 56

    training_label = pd.read_csv('input/train_labels.csv')
    category_index = {}
    current_index = 0
    for index, row in training_label.iterrows():
        if row['Category'] not in category_index:
            category_index[row['Category']] = current_index
            current_index += 1

    targets = []
    for index, row in training_label.iterrows():
        targets.append(category_index[row['Category']])
    targets = np.array(targets)

    training = np.load('input/train_images.npy', encoding='bytes')
    data = np.zeros(shape=(10000, IMG_SIZE*IMG_SIZE + 1), dtype=float)

    for i in range(10000):
        temp_img = cleanNoise3(training[i, 1])
        temp_img = TrimImage(temp_img)
        temp_img = temp_img.flatten()
        temp_img[0] = 10
        temp_img = np.divide(temp_img, 255)
        temp_img = np.append(temp_img, targets[i])
        data[i] = temp_img
    
    return data, targets

def predict(network, dataSet):
    correct = 0
    bad = 0
    predictions = list()
    for row in dataSet:
        outputs = feedForward(network, row)
        prediction = outputs.index(max(outputs))
        predictions.append(prediction)
        if (row[-1] == prediction):
            correct += 1
        else:
            bad += 1
    return predictions

def main():
        
    # Get the data and separate them into Training, Validation and Test sets
    data, targets = getData()

    # Training set
    trainingSet = data[0:int(len(data)*0.6)]

    # Test set
    testSet = data[int(len(data)*0.6):int(len(data)*0.8)]
    testTarget = targets[int(len(data)*0.6):int(len(data)*0.8)]

    # Validation set
    validationSet = data[int(len(data)*0.8):len(data)]
    validationTarget = targets[int(len(data)*0.8):len(data)]

    # Get necessary info to create the network
    numInputs = len(trainingSet[0]) - 1
    numClasses = len(set([row[-1] for row in trainingSet]))

    # Choose the number of layers, epochs and the learning rate of the network
    numLayers = 1
    numEpochs = 50
    learningRate = 0.5

    # Construct and train the Feed Foward Neural Network
    network = constructNet(numInputs, numLayers, numClasses)
    train(network, trainingSet, learningRate, numEpochs, numClasses)
    predictions = predict(network, validationSet)

    # Show the results
    print("Layers: {}, Epochs: {}, Learning Rate: {}".format(numLayers, numEpochs, learningRate))
    print("\tF1_Measure: {}".format(skMetrics.f1_score(testTarget, predictions,  average='micro')))

main()
