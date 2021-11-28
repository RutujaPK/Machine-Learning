from src.utils import softmax
import numpy as np


class NaiveBayes:
    """
    A Naive Bayes classifier for binary data.
    """

    def __init__(self, smoothing=1):
        """
        Args:
            smoothing: controls the smoothing behavior when computing p(x|y).
                If the word "jackpot" appears `k` times across all documents with
                label y=1, we will instead record `k + self.smoothing`. Then
                `p("jackpot" | y=1) = (k + self.smoothing) / Z`, where Z is a
                normalization constant that accounts for adding smoothing to
                all words.
        """
        self.smoothing = smoothing

    def predict(self, X):
        """
        Return the most probable label for each row x of X.
        You should not need to edit this function.
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        """
        Using self.p_y and self.p_x_y, compute the probability p(y | x) for each row x of X.
        While you will have used log probabilities internally, the returned array should be
            probabilities, not log probabilities. You may use src.utils.softmax to transform log
            probabilities to probabilities.

        Args:
            X: a data matrix of shape `[n_documents, vocab_size]` on which to predict p(y | x)

        Returns
            probs: an array of shape `[n_documents, n_labels]` where probs[i, j] contains
                the probability `p(y=j | X[i, :])`. Thus, for a given row of this array,
                sum(probs[i, :]) == 1.
        """
        n_docs, vocab_size = X.shape
        n_labels = 2

        assert hasattr(self, "p_y") and hasattr(
            self, "p_x_y"), "Model not fit!"
        assert vocab_size == self.vocab_size, "Vocab size mismatch"

        probs = np.ones((n_docs, n_labels))

        for i in range(n_docs):
            a = self.p_y[0]
            probability_1 = a
            probability_0 = np.log(1 - np.exp(a))

            for j in range(vocab_size):
                b = X[i, j]
                probability_1 += self.p_x_y[j, 1] * b
                probability_0 += self.p_x_y[j, 0] * b

            probs_array = softmax(
                np.array([probability_0, probability_1]), axis=1)

            for k in range(n_labels):
                probs[i, k] = probs_array[0, k]

        return probs

        # raise NotImplementedError

    def fit(self, X, y):
        """
        Compute self.p_y and self.p_x_y using the training data.
        You should store log probabilities to avoid underflow.
        This function *should not* use unlabeled data. Wherever y is NaN, that
        label and the corresponding row of X should be ignored.

        self.p_y should contain the marginal probability of each class label.
            Because we are doing binary classification, you may choose
            to represent p_y as a single value representing p(y=1)

        self.p_x_y should contain the conditional probability of each word
            given the class label: p(x | y). This should be an array of shape
            [n_vocab, n_labels].  Remember to use `self.smoothing` to smooth word counts!
            See __init__ for details. If we see M total words across all N documents with
            label y=1, have a vocabulary size of V words, and see the word "jackpot" `k`
            times, then: `p("jackpot" | y=1) = (k + self.smoothing) / (M + self.smoothing *
            V)` Note that `p("jackpot" | y=1) + p("jackpot" | y=0)` will not sum to 1;
            instead, `sum_j p(word_j | y=1)` will sum to 1.

        Args: X, a sparse matrix of word counts; Y, an array of labels
        Returns: None
        """

        # removing all unlabelled data from the given set of data values
        X = X[~np.isnan(y)]
        y = y[~np.isnan(y)]

        num_words_0 = np.sum(X[y == 0])
        num_words_1 = np.sum(X[y == 1])

        n_docs, vocab_size = X.shape
        n_labels = 2
        self.vocab_size = vocab_size

        # calculating marginal probability of each class
        self.p_y = np.array([np.log(np.size(y[y == 1]) / np.size(y))])
        # calculating conditional probability of each word given the class label: p(x | y)
        self.p_x_y = np.ones((vocab_size, n_labels))*(-1)

        '''
        If we see M total words across all N documents with
            label y=1, have a vocabulary size of V words, and see the word "jackpot" `k`
            times, then: `p("jackpot" | y=1) = (k + self.smoothing) / (M + self.smoothing *
            V)` Note that `p("jackpot" | y=1) + p("jackpot" | y=0)` will not sum to 1;
            instead, `sum_j p(word_j | y=1)` will sum to 1.
        '''

        for i in range(vocab_size):
            word_0 = np.sum(X[y == 0][:, i])
            if (word_0 + self.smoothing) != 0:
                self.p_x_y[i, 0] = np.log(
                    (word_0 + self.smoothing) / (num_words_0 + vocab_size * self.smoothing))

            word_1 = np.sum(X[y == 1][:, i])
            if (word_1 + self.smoothing) != 0:
                self.p_x_y[i][1] = np.log(
                    (word_1 + self.smoothing) / (num_words_1 + vocab_size * self.smoothing))

        self.p_x_y[np.where(self.p_x_y == -1)] = np.NINF

    def likelihood(self, X, y):
        """
        Using fit self.p_y and self.p_x_y, compute the log likelihood of the data.
            You should use logs to avoid underflow.
            This function should not use unlabeled data. Wherever y is NaN,
            that label and the corresponding row of X should be ignored.

        Recall that the log likelihood of the data can be written:
          `sum_i (log p(y_i) + sum_j log p(x_j | y_i))`

        Note: If the word w appears `k` times in a document, the term
            `p(w | y)` should appear `k` times in the likelihood for that document!

        Args: X, a sparse matrix of word counts; Y, an array of labels
        Returns: the (log) likelihood of the data.
        """
        assert hasattr(self, "p_y") and hasattr(
            self, "p_x_y"), "Model not fit!"

        likelihood_value = 0

        # removing all unlabelled data from the given set of data values
        X = X[~np.isnan(y)]
        y = y[~np.isnan(y)]

        n_docs, vocab_size = X.shape

        '''
        log likelihood of the data can be written:
          `sum_i (log p(y_i) + sum_j log p(x_j | y_i))`

        Therefore, if the word w appears `k` times in a document, the term
            `p(w | y)` should appear `k` times in the likelihood for that document!
        '''
        for i in range(n_docs):
            a = self.p_y[0]
            if not y[i]:
                likelihood_value += np.log(1 - np.exp(a))
            else:
                likelihood_value += a
            for j in range(vocab_size):
                if X[i, j] != 0:
                    likelihood_value += X[i, j] * self.p_x_y[j, int(y[i])]

        return likelihood_value


class NaiveBayesEM(NaiveBayes):
    """
    A NaiveBayes classifier for binary data,
        that uses unlabeled data in the Expectation-Maximization algorithm
    """

    def __init__(self, max_iter=10, smoothing=1):
        """
        Args:
            max_iter: the maximum number of iterations in the EM algorithm,
                where each iteration contains both an E step and M step.
                You should check for convergence after each iterations,
                e.g. with `np.isclose(prev_likelihood, likelihood)`, but
                should terminate after `max_iter` iterations regardless of
                convergence.
            smoothing: controls the smoothing behavior when computing p(x|y).
                If the word "jackpot" appears `k` times across all documents with
                label y=1, we will instead record `k + self.smoothing`. Then
                `p("jackpot" | y=1) = (k + self.smoothing) / Z`, where Z is a
                normalization constant that accounts for adding smoothing to
                all words.
        """
        self.max_iter = max_iter
        self.smoothing = smoothing

    def fit(self, X, y):
        """
        Compute self.p_y and self.p_x_y using the training data.
        You should store log probabilities to avoid underflow.
        This function *should* use unlabeled data within the EM algorithm.

        During the E-step, use the superclass self.predict_proba to
            infer a distribution over the labels for the unlabeled examples.
            Note: you should *NOT* replace the true labels with your predicted
            labels. You can use a `np.where` statement to only update the
            labels where `np.isnan(y)` is True.

        During the M-step, update self.p_y and self.p_x_y, similar to the
            `fit()` call from the NaiveBayes superclass. However, when counting
            words in an unlabeled example to compute p(x | y), instead of the
            binary label y you should use p(y | x).

        For help understanding the EM algorithm, refer to the lectures and
            http://www.cs.columbia.edu/~mcollins/em.pdf
            This PDF is also uploaded to the course website under readings.
            While Figure 1 of this PDF suggests randomly initializing
            p(y) and p(x | y) before your first E-step, please initialize
            all probabilities equally; e.g. if your vocab size is 4, p(x | y=1)
            would be 1/4 for all values of x. This will make it easier to
            debug your code without random variation, and will checked
            in the `test_em_initialization` test case.

        self.p_y should contain the marginal probability of each class label.
            Because we are doing binary classification, you may choose
            to represent p_y as a single value representing p(y=1)

        self.p_x_y should contain the conditional probability of each word
            given the class label: p(x | y). This should be an array of shape
            [n_vocab, n_labels].  Remember to use `self.smoothing` to smooth word counts!
            See __init__ for details. If we see M total
            words across all documents with label y=1, have a vocabulary size
            of V words, and see the word "jackpot" `k` times, then:
            `p("jackpot" | y=1) = (k + self.smoothing) / (M + self.smoothing * V)`
            Note that `p("jackpot" | y=1) + p("jackpot" | y=0)` will not sum to 1;
            instead, `sum_j p(word_j | y=1)` will sum to 1.

        Args: X, a sparse matrix of word counts; Y, an array of labels
        Returns: None
        """
        n_docs, vocab_size = X.shape
        n_labels = 2
        self.vocab_size = vocab_size

        # calculating marginal probability of each class
        self.p_y = np.array([np.log(0.5)])
        # calculating conditional probability of each word given the class label: p(x | y)
        self.p_x_y = np.ones((vocab_size, n_labels)) * np.log(1 / vocab_size)

        prev = np.inf
        temp = self.likelihood(X, y)

        '''
            If we see M total
            words across all documents with label y=1, have a vocabulary size
            of V words, and see the word "jackpot" `k` times, then:
            `p("jackpot" | y=1) = (k + self.smoothing) / (M + self.smoothing * V)`
            Note that `p("jackpot" | y=1) + p("jackpot" | y=0)` will not sum to 1;
            instead, `sum_j p(word_j | y=1)` will sum to 1.
        '''
        while self.max_iter and not np.isclose(prev, temp):

            y_predict = self.predict_proba(X)
            y_predict = np.where(np.isnan(y), y_predict[:, 1], y)
            self.p_y = np.array([np.log((1 / n_docs) * np.sum(y_predict))])
            a = y_predict

            m_0 = np.sum(X.toarray() * (1 - a).reshape((n_docs, 1)))
            m_1 = np.sum(X.toarray() * a.reshape((n_docs, 1)))

            for i in range(vocab_size):
                k_0 = np.sum(X.toarray()[:, i] * (1 - a))
                k_1 = np.sum(X.toarray()[:, i] * a)

                self.p_x_y[i][0] = np.log(
                    (k_0 + self.smoothing) / (m_0 + self.smoothing * vocab_size))
                self.p_x_y[i][1] = np.log(
                    (k_1 + self.smoothing) / (m_1 + self.smoothing * vocab_size))

            prev = temp
            temp = self.likelihood(X, y)

            self.max_iter -= 1

        return self.p_x_y

    def likelihood(self, X, y):
        """
        Using fit self.p_y and self.p_x_y, compute the likelihood of the data.
            You should use logs to avoid underflow.
            This function *should* use unlabeled data.

        For unlabeled data, we define `delta(y | i) = p(y | x_i)` using the
            previously-learned p(x|y) and p(y) necessary to compute
            that probability. For labeled data, we define `delta(y | i)`
            as 1 if `y_i = y` and 0 otherwise; this is because for labeled data,
            the probability that the ith example has label y_i is 1.
            Following http://www.cs.columbia.edu/~mcollins/em.pdf,
            the log likelihood of the data can be written as:

            `sum_i sum_y (delta(y | i) * (log p(y) + sum_j log p(x_{i,j} | y)))`

        Note: If the word w appears `k` times in a document, the term
            `p(w | y)` should appear `k` times in the likelihood for that document!

        Args: X, a sparse matrix of word counts; Y, an array of labels
        Returns: the (log) likelihood of the data.
        """

        assert hasattr(self, "p_y") and hasattr(
            self, "p_x_y"), "Model not fit!"

        n_docs, vocab_size = X.shape
        n_labels = 2

        likelihood_value = 0
        predict = self.predict_proba(X)
        y_predict = np.where(np.isnan(y), predict[:, 1], y)

        for i in range(n_docs):
            a = y_predict[i]
            b = self.p_y[0]
            likelihood_value += a * b + \
                (1 - a) * np.log(1 - np.exp(b))

            for j in range(vocab_size):
                likelihood_value += a * X[i, j] * self.p_x_y[j, 1]
                likelihood_value += (1 - a) * \
                    X[i, j] * self.p_x_y[j, 0]

        return likelihood_value
