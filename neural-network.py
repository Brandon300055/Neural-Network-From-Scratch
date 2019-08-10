import numpy as np

# sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * ( 1 - x )

# the feature we are training for is to spot if the first value is 1 or not
# training data
training_inputs = np.array([
    [0,0,1], [1,1,1], [1,0,1],[0,1,1],
                           ])
print("here")
print(training_inputs)

# data labels .T transpose it to make it a 4 by 1 matrix
training_output = np.array([[0,1,1,0]]).T

# now we assign random values to our weights/bias
# this will create our base line before we train

# seed the random with the seed of 1 part of np
(np.random.seed(1))

# list out our weights
synaptic_weights = 2 * np.random.random((3, 1)) - 1


print("Random Starting synaptic weights: ")
print(synaptic_weights)


# the node:

# construct a node by taking the randomly
# generated input weights and outputting them
# summing them all together and passing the
# output throw the sigmoid function
# np.dot mutiplys the vector of its two parameters
#
for i in range(20000):
    input_layer = training_inputs
    output = sigmoid(np.dot(input_layer, synaptic_weights))

    error = training_output - output

    adjustments = error * sigmoid_derivative(output)

    synaptic_weights += np.dot (input_layer.T, adjustments)

print('Synaptic Weights after training: ')
print(synaptic_weights)

print('Outputs after training: ')
print(output)

# now with the output of this data we will now
# calculate the error and adjust the weights accordingly
# this is to reduce the lose
# then we repeat these steps 20,000 times lol

# backproporgation is done with the Error Weight Derivative

# output - actual output
# input = either 1 or 0
# adjust weights by = error * input * sigmord` (output)

#  sigmord` (x) = x * ( 1 - x )


