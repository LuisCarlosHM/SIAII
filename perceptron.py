import numpy as np

class Perceptron:
    def __init__(self, num_inputs):
        self.weights = np.zeros(num_inputs)
        self.bias = 0

    def predict(self, inputs):
        activation = np.dot(inputs, self.weights) + self.bias
        return 1 if activation >= 0 else -1

    def train(self, training_inputs, labels, num_epochs):
        for _ in range(num_epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights += (label - prediction) * inputs
                self.bias += (label - prediction)

    def test(self, test_inputs, labels):
        correct_predictions = 0
        total_predictions = len(test_inputs)

        for inputs, label in zip(test_inputs, labels):
            prediction = self.predict(inputs)
            if prediction == label:
                correct_predictions += 1

        accuracy = correct_predictions / total_predictions
        return accuracy