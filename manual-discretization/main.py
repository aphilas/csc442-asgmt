import copy
import logging
import os
import traceback

import weka.core.jvm as jvm
from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random
from weka.core.converters import Loader
from weka.filters import Filter

# TODO: set logger level for jvm

def get_data_dir():
    """
    Returns the data directory.
    :return: the data directory
    :rtype: str
    """
    return os.path.dirname(os.path.abspath(__file__)) + os.sep + "data"


def load_dataset(dataset_name: str):
    """Returns a dataset given it's name in the Weka data directory (without .arff extension)"""
    file = get_data_dir() + os.sep + dataset_name + ".arff"
    loader = Loader("weka.core.converters.ArffLoader")
    dataset = loader.load_file(file)
    dataset.class_is_last()
    return dataset

def bin_dataset(dataset, attr_index, boundaries):
    options = ["-R", str(attr_index), "-V", "-E", f"ifelse(A>{boundaries[0]},ifelse(A>{boundaries[1]}, 3, 2), 1)"]
    bin_filter = Filter(classname="weka.filters.unsupervised.attribute.MathExpression", options=options)
    bin_filter.inputformat(dataset)
    binned = bin_filter.filter(dataset)

    nominal_filter = Filter(classname="weka.filters.unsupervised.attribute.NumericToNominal", options=["-R", str(attr_index)])
    nominal_filter.inputformat(binned)
    nominal = nominal_filter.filter(binned)

    return nominal
    
def main():
    dataset = load_dataset("chronic_kidney_disease_full")

    # Define binning boundaries
    # attr_name, index, boundaries
    # TODO: age? - 12, 17, 44, 64
    FILTERS = [
        ("wbcc", (4500, 11000)),
        ("rbcc", (4.2, 6.7)),
        ("bu", (7, 21)),
        ("hemo", (12.1, 17.2)),
        ("sc", (0.5, 1.3)),
        ("sod", (135, 145)),
        ("pot", (3.5, 5.5)),
    ]

    binned = copy.deepcopy(dataset)

    # Bin the data
    for (attr_name, boundaries) in FILTERS:
        # TODO: check for off-by-one error
        binned = bin_dataset(binned, binned.attribute_by_name(attr_name).index+1, boundaries)

    classifiers = [
        ("IBk", Classifier(classname="weka.classifiers.lazy.IBk")), # KNN?
        ("J48", Classifier(classname="weka.classifiers.trees.J48")),
        ("MLP", Classifier(classname="weka.classifiers.functions.MultilayerPerceptron")),
    ]

    print("classifier\tno binning\tbinning")

    for classifier_name, classifier in classifiers:
        print(classifier_name, end='\t')
        print(evaluate_classifier(dataset, classifier), end='\t')
        print(evaluate_classifier(binned, classifier))

def evaluate_classifier(dataset, classifier, random_seed=42):
    evaluation = Evaluation(dataset)
    evaluation.evaluate_train_test_split(classifier, dataset, 70, rnd=Random(random_seed), output=None)
    return evaluation.percent_correct


if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
