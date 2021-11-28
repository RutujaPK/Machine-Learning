# 7. (3 points) MNIST with a Neural Net
# With scikit-learn it is easy to train a multi-layer perceptron to classify the digits in the full MNIST dataset.
# Perform the following experiments. Don't forget to put clearly commented code for these experiments in the 'experiments' folder.

from your_code import GradientDescent, load_data
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

# Q7a.
# (1 point) Do an examination of how many nodes you should use in the hidden layer to learn the full 10-digit MNIST problem.
#  Use the defaults for all values except hidden_layer_sizes. Make 5 networks, each with a single hidden layer.
# Try each of the following hidden layer sizes: 1, 4, 16, 64, 256. Train all networks on the same training/test split.
# Get the accuracy. Repeat this 10 times. Report the mean and standard deviation of the accuracy of every network.

train_features, test_features, train_targets, test_targets = load_data(
    'mnist-multiclass', fraction=0.8)

accuracies = dict()
hidden_layer = [1, 4, 16, 64, 256]

for i in range(5):
    accuracies[i] = []

for i in range(10):
    for j in range(5):
        classifier = MLPClassifier(hidden_layer_sizes=(hidden_layer[j], 10))
        classifier.fit(train_features, train_targets)
        preds = classifier.predict(test_features)
        accuracies[j].append(np.mean(preds == test_targets))

mean_accuracy = []
std_accuracy = []

for i in accuracies.keys():
    mean_accuracy.append(np.mean(accuracies[i]))
    std_accuracy.append(np.std(accuracies[i]))

print(mean_accuracy)
print(std_accuracy)
best_hidden = hidden_layer[np.argmax(mean_accuracy)]
print("Best hidden layer value", best_hidden)

# Q7b.
# (1 point) Use the best network architecture from part A. Now we'll try varying the activation function.
# Perform the same experiment as in part A, but instead of varying architecture try each of the following
# activation functions {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}.

activation_functions = ['identity', 'logistic', 'tanh', 'relu']
activation_accuracies = dict()
for i in range(4):
    activation_accuracies[i] = []

for i in range(10):
    for j in range(4):
        classifier = MLPClassifier(hidden_layer_sizes=(
            best_hidden, 10), activation=activation_functions[j])
        classifier.fit(train_features, train_targets)
        preds = classifier.predict(test_features)
        activation_accuracies[j].append(np.mean(preds == test_targets))

mean_accuracy_activation = []
std_accuracy_activation = []

for i in activation_accuracies.keys():
    mean_accuracy_activation.append(np.mean(activation_accuracies[i]))
    std_accuracy_activation.append(np.std(activation_accuracies[i]))

print(mean_accuracy_activation)
print(std_accuracy_activation)

best_activation = activation_functions[np.argmax(mean_accuracy_activation)]

print("Best activation function", best_activation)

# Q7c.
# (1 point) Now that you've got the best network size and activation function, see if you can regularize it.
# Use your best hidden layer size. Use the best activation function. Perform the same experiment as in part A.
# but instead of varying architecture, vary the L2 regularization term (you'll have to find it in the docs) by
# powers of 10: {1,.1,.01,.001,.001}.

alpha_values = [1, 0.1, 0.01, 0.001, 0.0001]
alpha_accuracies = dict()

for i in range(5):
    alpha_accuracies[i] = []

for i in range(10):
    for j in range(5):
        classifier = MLPClassifier(hidden_layer_sizes=(
            best_hidden, 10), activation=best_activation, alpha=alpha_values[j])
        classifier.fit(train_features, train_targets)
        preds = classifier.predict(test_features)
        alpha_accuracies[j].append(np.mean(preds == test_targets))

mean_accuracy_alpha = []
std_accuracy_alpha = []
for i in alpha_accuracies.keys():
    mean_accuracy_alpha.append(np.mean(alpha_accuracies[i]))
    std_accuracy_alpha.append(np.std(alpha_accuracies[i]))

print(mean_accuracy_alpha)
print(std_accuracy_alpha)

best_alpha = alpha_values[np.argmax(mean_accuracy_alpha)]
print("Best alpha value", best_alpha)


# Q8a.
# (1 point) Pick the best network you found from question 7 and visualize its hidden layer.
# Go back and look at the links in question 7 if you don't know how to do that.
# best_hidden = 64
# best_activation = 'relu'
# best_alpha = 1

classifier = MLPClassifier(hidden_layer_sizes=(best_hidden, 10),
                           activation=best_activation, alpha=best_alpha)
classifier.fit(train_features, train_targets)

figures, axes = plt.subplots(4, 4)

v_minimum, v_maximum = classifier.coefs_[0].min(), classifier.coefs_[0].max()
for coef, ax in zip(classifier.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray,
               vmin=0.5 * v_minimum, vmax=0.5 * v_maximum)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()
