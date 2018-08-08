import numpy as np
import math
from itertools import product

def gini_coefficient(values):
    gini_numerator = np.sum( np.sum(np.abs(values - val)) for val in values )
    gini_denominator = 2 * len(values) * np.sum(values)
    return gini_numerator / gini_denominator

def theil_index(values, adjust_for_neg=False):
    mean = np.mean(values)

    values = [abs(v) + (abs(mean) if v * mean < 0 else 0) for v in values]
    mean = abs(mean)

    theil = sum((v / mean * np.log(v / mean) if v != 0 \
            else 0.) for v in values)
    return theil / len(values)

def ge_2_index(values):
    mean = np.mean(values)
    if mean == 0:
        print("Warning: 0 benefit mean in GE_2 computation")
        return 0.
    ge_2 = np.sum(np.square(values / mean) - 1)
    ge_2 /= (len(values) * 2)
    return ge_2

def decompose_inequality(group_benefits):
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
        print("g_benefits:", g_benefits)
        sub_mean_util = np.mean(g_benefits)
        means_sqr = np.square(sub_mean_util / mean_util)
        sub_inequality_weight = (len(g_benefits) / len(overall_benefits)) * means_sqr

        sub_inequality = ge_2_index(g_benefits)
        sub_inequalities[group] = sub_inequality_weight * sub_inequality
        unweighted_inequalities[group] = sub_inequality

        intergroup_component = len(g_benefits) / (2 * len(overall_benefits)) * \
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
