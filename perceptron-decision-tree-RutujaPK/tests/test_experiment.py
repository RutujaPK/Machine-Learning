import numpy as np
from src import run
import os
import json

datasets = [
    os.path.join('data', x)
    for x in os.listdir('data')
    if x.endswith('.csv')
]

def xp_dataset_name(key):
    dataset = [d for d in datasets if key in d]
    if not dataset:
        raise ValueError('Dataset ' + key + ' cannot be found')
    return dataset[0]


def test_information_gain():
    from src import load_data
    from src import information_gain

    _features, _targets, _attribute_names = load_data('data/PlayTennis.csv')
    iGHumidity = information_gain(_features, 2, _targets)
    iGWind = information_gain(_features, 3, _targets)
    realIGHumidity = 0.1515
    realIGWind = 0.048

    assert np.abs(iGHumidity-realIGHumidity)< 1e-3
    assert np.abs(iGWind - realIGWind) < 1e-3

def test_experiment_run_decision_tree():
    accuracies = {}
    for data_path in datasets:
        learner_type = 'decision_tree'
        confusion_matrix, accuracy, precision, recall, f1_measure = (
            run(data_path, learner_type, 1.0)
        )
        accuracies[data_path] = accuracy

    accuracy_goals = {
        xp_dataset_name('ivy-league.csv'): .95,
        xp_dataset_name('xor.csv'): 1.0,
        xp_dataset_name('candy-data.csv'): .75,
        xp_dataset_name('majority-rule.csv'): 1.0,
        xp_dataset_name('circles.csv'): 0.8,
        xp_dataset_name('blobs.csv'): 0.8,
    }
    for key in accuracy_goals:
        assert (accuracies[key] >= accuracy_goals[key]), key


def test_experiment_run_perceptron():
    accuracies = {}
    for data_path in datasets:
        learner_type = 'perceptron'
        confusion_matrix, accuracy, precision, recall, f1_measure = (
            run(data_path, learner_type, 1.0)
        )
        accuracies[data_path] = accuracy

    accuracy_goals = {
        xp_dataset_name('ivy-league.csv'): .85,
        xp_dataset_name('candy-data.csv'): .6,
        xp_dataset_name('majority-rule.csv'): 1.0,
        xp_dataset_name('blobs.csv'): 0.9,
        xp_dataset_name('circles.csv'): 0.54,
        xp_dataset_name('parallel_lines.csv'): 1.0,
    }
    for key in accuracy_goals:
        assert (accuracies[key] >= accuracy_goals[key]), key

def test_experiment_run_prior_probability():
    accuracies = {}
    for data_path in datasets:
        learner_type = 'prior_probability'
        confusion_matrix, accuracy, precision, recall, f1_measure = (
            run(data_path, learner_type, 1.0)
        )
        accuracies[data_path] = accuracy
    dataset = xp_dataset_name('ivy-league.csv')
    assert (accuracies[dataset] > .2)

def test_experiment_run_and_compare():
    for data_path in datasets:
        accuracies = {}
        learner_types = ['prior_probability', 'decision_tree']
        for learner_type in learner_types:
            accuracies[learner_type] = run(data_path, learner_type, 1.0)[1]
        if 'candy' in data_path or 'ivy' in data_path:
            assert (
                accuracies['decision_tree'] > accuracies['prior_probability']
            )
