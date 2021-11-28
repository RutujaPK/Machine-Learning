# Coding (12 points)
Your task is to implement machine learning algorithms:

1. Prior probability (in `src/prior_probability.py`)
2. Decision tree (in `src/decision_tree.py`)
3. Perceptron (in `src/perceptron.py`)

You will also write code that reads in data into numpy arrays and code that manipulates
data for training and testing in `code/data.py`.

You will implement evaluation measures in `code/metrics.py`:

1. Confusion matrix (`code/metrics.py -> compute_confusion_matrix`)
2. Precision and recall (`code/metrics.py -> compute_precision_and_recall`)
3. F1-Measure (`code/metrics.py -> compute_f1_measure`)

The entire workflow will be encapsulated in `code/experiment.py -> run`. The run function 
will allow you to run each approach on different datasets easily. You will have to 
implement this `run` function.

Your goal is to pass the test suite (contained in `tests/`). Once the tests are passed, you 
may move on to the next part - reporting your results.

Your grade for this section is defined by the autograder. If it says you got an 80/100,
you get 9.6/12 points here. While there are 12 tests listed below, not all tests are worth equal points. Suggested order for passing test_cases:

1.  test_load_data
2.  test_train_test_split
3.  test_confusion_matrix
4.  test_accuracy
5.  test_precision_and_recall
6.  test_f1_measure
7.  test_experiment_run_prior_probability
8.  test_perceptron
9.  test_experiment_run_perceptron
10. test_information_gain
11. test_experiment_run_decision_tree
12. test_experiment_run_and_compare

# Free-response Questions (8 points)

To answer some of these questions, you will have to write extra code (that is not covered by the test cases). The extra code should import your implementation and run experiments on the various datasets (e.g., choosing `ivy-league.csv` for a dataset and doing `experiment.run` with an 80/20 train/test split). You may use the visualization code provided in `src/visualize.py`.

**You do not need to submit extra code you write to answer these questions.**

1. In the coding section of this assignment, you trained Decision Tree and Perceptron models on several datasets. For each dataset in the `data` directory, do the following:

   * Generate __5__ random train/test splits of the dataset using `fraction = 0.8`
   * For each split, train Decision Tree and Perceptron models on the training portion of the data
   * For each split, compute the accuracy of each model on the testing portion of the data
   * Report the average test accuracy of each model type across the __5__ splits
   * For Decision Tree models, report the average number of nodes in the tree, and the average maximum depth (number of levels) of the tree
   * For Perceptron models, report if (and after how many iterations, on average) they converged
   
   You may find the code in `tests/test_experiment.py` helpful for getting started.
 
2. For these five datasets (`blobs.csv`, `circles.csv`, `crossing.csv`, `parallel_lines.csv`, and `transform_me.csv`):
   * Train a Perceptron model on the __entire dataset__ (`fraction=1.0`) and plot the resulting decision regions using the `plot_decision_regions` function provided in `visualize.py`
   * Train a Decision Tree model on the __entire dataset__ (`fraction=1.0`) and plot the resulting decision regions using the `plot_decision_regions` function provided in `visualize.py`
   
   For this question, __do not__ apply any transformation to the `transform_me` dataset. You should label your plots clearly using the `title` argument of the plotting function.   

3. For the Perceptron models you trained on the `circles` and (untransformed) `transform_me` datasets in __Question 2__, compute the precision, recall, and F1 scores over each dataset. While these datasets should look very similar when plotted, the performance of your Perceptron model as measured by these metrics should differ noticeably between the two. Explain why you think this is the case.

4. In `src/perceptron.py`, you passed a test case by implementing the function `transform_data`. Describe the transformation you made to the input data to pass the test case. Explain how it allowed you to pass the test case (i.e. how did the change made to the data enable the system to do what it could not previously do?).

5. Choose two datasets D1 and D2 such that Decision Tree has better accuracy than Perceptron on D1, but Perceptron has better accuracy than Decision Tree on D2. Give the names of your chosen datasets and compute accuracy with `fraction=1.0` in `train_test_split()`. For each dataset, visualize the data and explain why one model outperformed the other. You may reuse your code and/or scatter plots from __Question 2__. Your explanations should discuss the inductive biases of both models.

6. For each of the datasets you considered in the previous question, describe a learning algorithm adjustment or data transformation that would allow the worse-performing model to perfectly classify the data. Explain your decisions.

7. Assume you have a deterministic function that takes a fixed, finite number of Boolean inputs and returns a Boolean output. Can a Decision Tree be built to represent any such function? Give a simple proof or explanation for your answer. If you choose to give a proof, don't worry about coming up with a very formal or mathematical proof. It is up to you what you want to present as a proof.

8. There is a difference between a data structure (e.g. a decision tree) and the algorithm used to build that data structure (e.g. ID3). Different algorithms have different assumptions and biases built into them. At every decision point, the ID3 algorithm asks, "Which feature is most correlated with the output of the function we seek to model?" It then chooses to split on that feature. What is the inherent assumption built into using that question to choose splits? Describe a function with a Boolean output that violates this assumption but that could still be represented as a decision tree. Explain why it violates the assumption. Hint: keep track of the number of nodes in each decision tree you fit; which datasets require the largest trees? Why?
