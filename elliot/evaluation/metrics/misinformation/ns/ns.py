__version__ = '0.0.1'
__author__ = 'Benedikt Hornig'
__email__ = 'benedikthornig@outlook.com'

import pandas as pd

from elliot.evaluation.metrics.base_metric import BaseMetric

import numpy as np


class NS(BaseMetric):
    def __init__(self, recommendations: dict[int, list[tuple[int, float]]], config, params, eval_objects):
        """
        Constructor
        :param recommendations: list of recommendations in the form {user: [(item1,value1),...]}
        :param config: SimpleNameSpace that represents the configuration of the experiment
        :param params: Parameters of the model
        :param eval_objects: list of objects that may be useful for the computation of the different metrics
        """
        super().__init__(recommendations, config, params, eval_objects)
        self._cutoff = self._evaluation_objects.cutoff
        self._misinformation_labels = pd.read_csv(config.data_config.root_folder
                                                  + '/../../predictions_with_item_id_mapping.tsv',
                                                  sep='\t').set_index('video_id')
        self._misinformation_labels = self._misinformation_labels['label'].replace({
            'neutral': 0,
            'debunking': -1,
            'promoting': 1
        })

    @staticmethod
    def name():
        """
        Metric Name Getter
        :return: returns the public name of the metric
        """
        return "NS"

    def eval_user_metric(self):
        """
        Evaluation function
        :return: the overall averaged value of Normalized Score per user
        """
        return {user: self.__user_NS(user_recommendations, self._cutoff)
                for user, user_recommendations in self._recommendations.items()}

    def __user_NS(self, user_recommendations, cutoff):
        """
        Per User Normalized Score
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param cutoff: numerical threshold to limit the recommendation list
        :return: the value of the NS metric for the specific user
        """
        return self._misinformation_labels[[item[0] for item in user_recommendations[:cutoff]]].mean()
