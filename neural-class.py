import numpy as np

class NeuralNetwork():

    def __init__(self):

        # seed for random variables
        np.random.seed(1)

        # list out our weights
        # create random vaules for the weights
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    # our activation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # the deferential of our activation function
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # method will train our node
    def train(self, training_inputs, training_outputs, training_iterations):

        # loop for the number of retraining iterations
        for iteration in range(training_iterations):

            output = self.think(training_inputs)

            # determine the error and reduce the loss
            error = training_outputs - output

            # determine necessary adjustment to reduce the loss aka turn the nobe
            # note this is our backpropagation
            adjustments = np.dot(training_outputs.T, error * self.sigmoid_derivative(output))

            self.synaptic_weights += adjustments

    # Given that the weights are the most important bit the already train model is
    # the synaptic synaptic_weights. With that we can pass to new data to test
    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))

        return output


if __name__ == "__main__":
    neural_network = NeuralNetwork()

    print("Random starting synaptic weights: ")
    print(neural_network.synaptic_weights)

    # the feature we are training for is to spot if the first value is 1 or not
    # training data
    training_inputs = np.array([
        [0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1],
    ])

    # data labels .T transpose it to make it a 4 by 1 matrix
    training_output = np.array([[0, 1, 1, 0]]).T

    neural_network.train(training_inputs, training_output, 20000)

    print('Synaptic Weights after training: ')
    print(neural_network.synaptic_weights)

    # ask user for custom inputs
    # A = str(input("Input 1: "))
    # B = str(input("Input 2: "))
    # C = str(input("Input 3: "))

    # print("New situational data: ", A, B, C)
    # print("Output Data: ")
    # print(neural_network.think(np.array([A, B, C])))
    print("Output Data: ")
    print(neural_network.think(np.array([0, 0, 0])))

