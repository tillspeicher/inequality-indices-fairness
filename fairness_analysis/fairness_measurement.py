import numpy as np
import pandas as pd
import math
from itertools import product
from typing import Dict, TypeVar
from collections import Iterable

class InequalityFeatureSet:
    
    __T = TypeVar('__T', pd.DataFrame, dict)
    __feature_set = None
    
    def __init__(self, data: __T):
        if isinstance(data, pd.DataFrame):
            self.__features_from_df(data)
        elif isinstance(data, dict):
            self.__features_from_dict(data)
        else:
            raise ValueError("Trying to pass unsupported data type '{}' to an InequalityFeatureSet instance".format(type(data)))
            
        self.__decompose()
    
    def __decompose(self):
        inequality_decomposer = _InequalityDecomposer()
        for feature, attributes in self.__feature_set.items():
            print("[{}] {}".format(feature, inequality_decomposer.run(attributes)))
    
    def __features_from_df(self, dataframe: pd.DataFrame) -> None:
        if len(dataframe.columns) != len(set(dataframe.columns)):
            raise ValueError("Trying to pass DataFrame with duplicate columns to an InequalityFeatureSet instance")
        if 'benefit' not in dataframe.columns:
            raise ValueError("Trying to pass DataFrame without a 'benefit' column to an InequalityFeatureSet instance")
            
        self.__feature_set = {
            feature: dataframe.groupby(feature)['benefit'].agg(list).to_dict() for feature in dataframe.columns if feature != 'benefit'
        }
        
    def __features_from_dict(self, dictionary: Dict) -> None:
        if not all(isinstance(feature, str) for feature in dictionary.keys()):
            raise ValueError("Trying to pass non-string-formatted feature labels to and InequalityFeatureSet instance")
        if not all(isinstance(attributes, dict) for attributes in dictionary.values()):
            raise ValueError("Trying to pass non-dictionary-formatted feature attributes to an InequalityFeatureSet instance")
        if not all(all(isinstance(attribute_label, str) and isinstance(attribute_benefits, Iterable) for attribute_label, attribute_benefits in attributes.items()) for attributes in dictionary.values()):
            raise ValueError("Trying to pass ill-formatted attribute benefits to an InequalityFeatureSet instance")
        
        self.__feature_set = dictionary

class _InequalityDecomposer:

    def __ge_2_index(self, values: Iterable, alpha: float) -> float:
        mean = np.mean(values)
        if mean == 0:
            print("Warning: 0 benefit mean in GE_2 computation")
            return 0.
        ge_2 = np.sum(np.power(values / mean, alpha) - 1)
        ge_2 /= (len(values) * alpha * (alpha - 1))
        return ge_2

    def run(self, group_benefits: Dict[str, Iterable], alpha: float = 2.0) -> Dict:
        """
        Expects the benefits for each member of each group as a mapping
        of the form {'<group_name>': [benefit_user_1, benefit_user_2, ...]}

        Computes the decomposition of population benefits for the GE_2 index (which is subgroup-decomposable) into
        - overall inequality
        - between-group inequality
        - within-group inequality for each group as a mapping
            {'<group_name>': within-group inequality})
        - unweighted intermediate components
        """
        overall_benefits = np.concatenate(list(group_benefits.values()))
        overall_ineq = self.__ge_2_index(overall_benefits, alpha)
        mean_util = np.mean(overall_benefits)

        sub_inequalities = {}
        unweighted_inequalities = {}
        intergroup_components = {}
        intergroup_inequality = 0
        for group, g_benefits in group_benefits.items():
            sub_mean_util = np.mean(g_benefits)
            means_sqr = np.power(sub_mean_util / mean_util, alpha)
            sub_inequality_weight = (len(g_benefits) / len(overall_benefits)) * means_sqr

            sub_inequality = self.__ge_2_index(g_benefits, alpha)
            sub_inequalities[group] = sub_inequality_weight * sub_inequality
            unweighted_inequalities[group] = sub_inequality

            intergroup_component = len(g_benefits) / (len(overall_benefits) * alpha * (alpha - 1)) * \
                    (means_sqr - 1)
            intergroup_inequality += intergroup_component
            intergroup_components[group] = intergroup_component

        decomposition_sum = sum(sub_inequalities.values()) + intergroup_inequality
        assert math.isclose(overall_ineq, decomposition_sum, rel_tol=0.1), "overall inequality {} and decomposition sum {} not close".format(overall_ineq, decomposition_sum)

        return {
            'overall_ineq': overall_ineq,
            'between_ineq': intergroup_inequality,
            'within_ineqs': sub_inequalities,
            'unweighted_ineqs': unweighted_inequalities,
            'intergroup_components': intergroup_components
        }

    def __rank_between_group_inequality(self, benefits: Iterable, feature_groups: Iterable) -> Dict:
        """
        Ranks features based on the between-group inequality in the
        benefit distribution that they cause when the population
        is split according to them.

        Takes as arguments
        - the benefits for the entire population
        - a mapping from feature names to mappings from feature values to users that have the corresponding value, as numpy index or boolean arrays

        Returns
        - the feature names ranked by between-group inequality in ascending order
        - the between-group inequality for each feature
        - the within-group inequality for each value of each feature
        """

        feature_intergroup_inequalities = {}
        feature_withingroup_inequalities = {}
        for fname, groups in feature_groups.items():
            group_benefits = {gname: benefits[group] for gname, group in groups.items()}
            inequality_decomp = decompose_inequality(group_benefits)
            feature_intergroup_inequalities[fname] = inequality_decomp['between_ineq']
            feature_withingroup_inequalities[fname] = inequality_decomp['within_ineqs']

        ranked_features = sorted(feature_intergroup_inequalities, key=feature_intergroup_inequalities.get)
        return ranked_features, feature_intergroup_inequalities, feature_withingroup_inequalities
