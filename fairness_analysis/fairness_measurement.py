import numpy as np
import math
from itertools import product
from typing import Dict
from collections import Iterable

class InequalityDecomposer:

    def __init__(self):
        return

    @typechecked
    def __gini_coefficient(values: Iterable) -> float:
        gini_numerator = np.sum( np.sum(np.abs(values - val)) for val in values )
        gini_denominator = 2 * len(values) * np.sum(values)
        return gini_numerator / gini_denominator

    @typechecked
    def __theil_index(values: Iterable, adjust_for_neg: bool = False) -> float:
        mean = np.mean(values)

        values = [abs(v) + (abs(mean) if v * mean < 0 else 0) for v in values]
        mean = abs(mean)

        theil = sum((v / mean * np.log(v / mean) if v != 0 \
                else 0.) for v in values)
        return theil / len(values)

    @typechecked
    def __ge_2_index(values: Iterable, alpha: float) -> float:
        mean = np.mean(values)
        if mean == 0:
            print("Warning: 0 benefit mean in GE_2 computation")
            return 0.
        ge_2 = np.sum(np.power(values / mean, alpha) - 1)
        ge_2 /= (len(values) * alpha * (alpha - 1))
        return ge_2

    @typechecked
    def decompose(group_benefits: Dict[str, Iterable], alpha: float = 2.0) -> Dict:
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
        overall_ineq = ge_2_index(overall_benefits)
        mean_util = np.mean(overall_benefits)

        sub_inequalities = {}
        unweighted_inequalities = {}
        intergroup_components = {}
        intergroup_inequality = 0
        for group, g_benefits in group_benefits.items():
            sub_mean_util = np.mean(g_benefits)
            means_sqr = np.power(sub_mean_util / mean_util, alpha)
            sub_inequality_weight = (len(g_benefits) / len(overall_benefits)) * means_sqr

            sub_inequality = ge_2_index(g_benefits)
            sub_inequalities[group] = sub_inequality_weight * sub_inequality
            unweighted_inequalities[group] = sub_inequality

            intergroup_component = len(g_benefits) / (len(overall_benefits) * alpha * (alpha - 1)) * \
                    (means_sqr - 1)
            intergroup_inequality += intergroup_component
            intergroup_components[group] = intergroup_component

        decomposition_sum = sum(sub_inequalities.values()) + intergroup_inequality
        assert math.isclose(overall_ineq, decomposition_sum, rel_tol=0.1), "overall inequality {} and decomposition sum {} not close".format(overall_ineq, decomposition_sum)

        return {'overall_ineq': overall_ineq,
                'between_ineq': intergroup_inequality,
                'within_ineqs': sub_inequalities,
                'unweighted_ineqs': unweighted_inequalities,
                'intergroup_components': intergroup_components}

    @typechecked
    def __rank_between_group_inequality(benefits: Iterable, feature_groups: Iterable) -> Dict:
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
