import os
import traceback
from itertools import combinations
from typing import Generator, List, Set, Tuple

import weka.core.jvm as jvm
from weka.classifiers import Classifier, Evaluation, PredictionOutput
from weka.core.classes import Random
from weka.core.converters import Loader
from weka.filters import Filter

# cross-fold validation number of folds
FOLDS = 5


def get_data_dir():
    """
    Returns the data directory.
    :return: the data directory
    :rtype: str
    """
    rootdir = os.path.dirname(__file__)
    libdir = rootdir + os.sep + "data"
    return libdir


def load_dataset(dataset_name: str):
    """Returns a dataset given it's name in the Weka data directory (without .arff extension)"""
    file = get_data_dir() + os.sep + dataset_name + ".arff"
    loader = Loader("weka.core.converters.ArffLoader")
    dataset = loader.load_file(file)
    dataset.class_is_last()
    return dataset_name, dataset


def main():
    datasets = map(load_dataset, ["iris", "diabetes", "glass"])

    classifiers = [
        ("NaiveBayes", Classifier(classname="weka.classifiers.bayes.NaiveBayes")),
        ("IBk", Classifier(classname="weka.classifiers.lazy.IBk")),
    ]

    for dataset_name, dataset in datasets:
        print(dataset_name, "\n")

        for classifier_name, classifier in classifiers:
            print(classifier_name)
            exhaustive_selection(dataset, classifier)

        print("")


def exhaustive_selection(dataset, classifier: Classifier, random_seed=42):
    attributes = dataset.attribute_names()

    for subset, filter in generate_filters(dataset.num_attributes - 1):
        remove = Filter(
            classname="weka.filters.unsupervised.attribute.Remove",
            options=["-R", filter],
        )
        remove.inputformat(dataset)
        filtered = remove.filter(dataset)
        evaluation = Evaluation(filtered)
        evaluation.crossvalidate_model(
            classifier, filtered, FOLDS, Random(random_seed), output=None
        )

        print(
            ", ".join(map(lambda i: attributes[i], subset))
            + "\t"
            + str(evaluation.percent_correct)
        )


def filter_str(combination: Set[int]) -> str:
    # use Weka's 1-based indexing
    return ",".join(map(lambda v: str(v + 1), combination))


def generate_filters(n) -> List[Tuple[Tuple[int, ...], str]]:
    """Returns a zip of all combinations of range(0,n) and the respective filter strings for weka"""

    powerset = (c for k in range(n) for c in combinations(range(n), k + 1))
    every = set(range(n))
    filters = (every - set(c) for c in powerset)

    return list(zip(powerset, map(filter_str, filters)))


if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
