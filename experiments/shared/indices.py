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

def evaluate_inequality(diffs, use_indices, transformations, min_diff=None):
    """ Computes inequality for multiple distances and transformation functions used to make the benefit values positive """
    trans_diffs = {}
    for transform in transformations:
        if transform == 'none':
            trans_diffs['raw'] = diffs
        elif transform == 'shift':
            shifted_diffs = diffs - min_diff if min_diff < 0 else diffs
            trans_diffs['shifted'] = shifted_diffs
        elif transform == 'sigmoid':
            sigmoid_diffs = 1 / (1 + np.exp(-diffs))
            trans_diffs['sigmoid'] = sigmoid_diffs
        else:
            assert False, "Unrecognized transformation {}".format(transform)

    ineq_indices = {'Gini': gini_coefficient,
            'Theil': theil_index,
            'GE_2': ge_2_index}
    indices = {}
    for index in use_indices:
        indices[index] = ineq_indices[index]

    result = []
    col_names = []
    for (transform, diff), (index, index_func) in product(
            trans_diffs.items(), indices.items()):
        ind_val = index_func(diff)
        result.append(index_func(diff))
        col_names.append("{} ({})".format(index, transform))

    return result, col_names

def decompose_inequality(group_benefits):
    """
    Computes the decomposition of population benefits for the GE_2 index (which is subgroup-decomposable) into
    - overall inequality
    - between-group inequality
    - within-group inequality for each group
    - unweighted intermediate components
    """
    overall_benefits = np.concatenate(group_benefits)
    overall_ineq = ge_2_index(overall_benefits)
    mean_util = np.mean(overall_benefits)

    sub_inequalities = []
    unweighted_inequalities = []
    intergroup_components = []
    intergroup_inequality = 0
    for i, sub_ut in enumerate(group_benefits):
        sub_mean_util = np.mean(sub_ut)
        means_sqr = np.square(sub_mean_util / mean_util)
        sub_inequality_weight = (len(sub_ut) / len(overall_benefits)) * means_sqr

        sub_inequality = ge_2_index(sub_ut)
        sub_inequalities.append(sub_inequality_weight * sub_inequality)
        unweighted_inequalities.append(sub_inequality)

        intergroup_component = len(sub_ut) / (2 * len(overall_benefits)) * \
                (means_sqr - 1)
        intergroup_inequality += intergroup_component
        intergroup_components.append(intergroup_component)

    decomposition_sum = sum(sub_inequalities) + intergroup_inequality
    assert math.isclose(overall_ineq, decomposition_sum, rel_tol=0.1), "overall inequality {} and decomposition sum {} not close".format(overall_ineq, decomposition_sum)

    return {'overall_ineq': overall_ineq,
            'intergroup_ineq': intergroup_inequality,
            'subgroup_ineqs': sub_inequalities,
            'unweighted_ineqs': unweighted_inequalities,
            'intergroup_components': intergroup_components}

def compute_benefits(predictions, labels, binary=False):
    """ Converts predictions and labels into positive benefits """
    if binary:
        return predictions
    else:
        return (predictions - labels) + 1
