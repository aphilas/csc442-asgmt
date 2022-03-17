import os
import traceback
from itertools import combinations

import weka.core.jvm as jvm
from weka.classifiers import Classifier, Evaluation, PredictionOutput
from weka.core.classes import Random
from weka.core.converters import Loader
from weka.filters import Filter


def get_data_dir():
    """
    Returns the data directory.
    :return: the data directory
    :rtype: str
    """
    rootdir = os.path.dirname(__file__)
    libdir = rootdir + os.sep + "data"
    return libdir


def main():
    # load dataset
    iris_file = get_data_dir() + os.sep + "iris.arff"
    loader = Loader("weka.core.converters.ArffLoader")
    iris_data = loader.load_file(iris_file)
    iris_data.class_is_last()

    #! NOTE: 1-based indexing
    remove_attributes = (1,)

    # filter attributes
    remove = Filter(
        classname="weka.filters.unsupervised.attribute.Remove",
        options=["-R", ",".join(map(str, remove_attributes))],
    )
    remove.inputformat(iris_data)
    filtered_iris = remove.filter(iris_data)

    # evaluate
    classifier = Classifier(classname="weka.classifiers.bayes.NaiveBayes")
    evaluation = Evaluation(filtered_iris)
    pred_output = PredictionOutput(
        classname="weka.classifiers.evaluation.output.prediction.PlainText",
        options=["-distribution"],
    )
    evaluation.crossvalidate_model(
        classifier, filtered_iris, 5, Random(42), output=pred_output
    )
    print(evaluation.percent_correct)


def filter_str(c):
    return ",".join(map(lambda v: str(v + 1), c))


#! incomplete
def subsets(n_features):
    """Returns a list of all combinatons of range(0,n_features) and the respective filter strings for weka

    Args:
        n_features (int): number of features

    Returns:
        (list, list)
    """

    powerset = (
        combination
        for k in range(n_features)
        for combination in combinations(range(n_features), k + 1)
    )

    all = set(range(0, n_features))

    filters = (
        all - set(combination) for combination in powerset if all != set(combination)
    )

    return list(map(lambda c: list(c), powerset)), list(map(filter_str, filters))


if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
