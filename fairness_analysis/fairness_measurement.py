import math
import warnings
import numpy as np
import pandas as pd
from itertools import product
from collections import Iterable
from typing import Dict, Any, Callable, TypeVar

class InequalityFeatureSet:
    '''
    This class is responsible for calculating and formatting
    the inequality decomposition of a set of user-defined
    features.
    '''

    __T = TypeVar('__T', pd.DataFrame, dict)
    __feature_set = None
    __feature_set_decomposition = None
    
    def __init__(self, data: __T):
        '''
        The constructor takes care of initialializing the
        user-defined feature set in a standardized way, 
        initiating the decomposition process and storing
        those intermediate results.
        
        Args:
        data (__T): This parameter will only be valid if it is
            provided either as a pandas DataFrame (each column
            containing a categorical variable and an extra 
            column named 'benefit', with a pre-calculated
            benefit for every sample) or as a nested dictionary
            (in the form of {<feature>:{<attr>:[x1, ..., xn]}},
            where it can contain as many attributes for each 
            feature and as many features as provided)
        '''
        
        if isinstance(data, pd.DataFrame):
            self.__features_from_df(data)
        elif isinstance(data, dict):
            self.__features_from_dict(data)
        else:
            raise ValueError("Trying to pass unsupported data type '{}' to an InequalityFeatureSet instance".format(type(data)))
            
        self.__decompose()
    
    def get_decomposition(self, formatter: Callable = None) -> Any:
        '''
        This method returns an (otionally formatted) inequality
        decomposition for a set of user-defined features.
        
        Args:
        formatter (Callable): User-defined function to customize
            the format of the default decomposition dictionary
            
        Returns:
        Any: By default it will return a dictionary but, if provided
            a custom formatter, it can return any type of object
        '''
        
        if formatter is None:
            return self.__feature_set_decomposition
        elif not callable(formatter):
            raise ValueError("Trying to pass a non-callable formatter to an InequalityFeatureSet isntance")
        return formatter(self.__feature_set_decomposition)
    
    def __decompose(self) -> None:
        '''
        This method takes care of calculating the specified features'
        inequality decomposition and storing the final results on
        a private variable.
        '''
        
        inequality_decomposer = _InequalityDecomposer()
        self.__feature_set_decomposition = {
            feature: inequality_decomposer.run(attributes) for feature, attributes in self.__feature_set.items()
        }
    
    def __features_from_df(self, dataframe: pd.DataFrame) -> None:
        '''
        This method handles standardization of features formatted
        as pandas DataFrames.
        
        Args:
        dataframe (pd.DataFrame): dataframe-formatted features, without
            duplicate columns and with a compulsory column named 'benefit'
            (containing pre-calculated benefits for every sample)
        '''
        
        if len(dataframe.columns) != len(set(dataframe.columns)):
            raise ValueError("Trying to pass DataFrame with duplicate columns to an InequalityFeatureSet instance")
        if 'benefit' not in dataframe.columns:
            raise ValueError("Trying to pass DataFrame without a 'benefit' column to an InequalityFeatureSet instance")
            
        self.__feature_set = {
            feature: dataframe.groupby(feature)['benefit'].agg(list).to_dict() for feature in dataframe.columns if feature != 'benefit'
        }
        
    def __features_from_dict(self, dictionary: Dict) -> None:
        '''
        This method handles standardization of features formatted
        as nested dictionaries.
        
        Args:
        dictionary (Dict): nested dictionary of features, in the form
            of {<feature>:{<attr>:[x1, ..., xn]}}, where there can be
            as many attributes and features as necessary (to accomodate
            for multiple features and multi-dimentional benefits)
        '''
        
        if not all(isinstance(feature, str) for feature in dictionary.keys()):
            raise ValueError("Trying to pass non-string-formatted feature labels to and InequalityFeatureSet instance")
        if not all(isinstance(attributes, dict) for attributes in dictionary.values()):
            raise ValueError("Trying to pass non-dictionary-formatted feature attributes to an InequalityFeatureSet instance")
        if not all(all(isinstance(attribute_label, str) and isinstance(attribute_benefits, Iterable) for attribute_label, attribute_benefits in attributes.items()) for attributes in dictionary.values()):
            raise ValueError("Trying to pass ill-formatted attribute benefits to an InequalityFeatureSet instance")
        
        self.__feature_set = dictionary

class _InequalityDecomposer:
    '''
    This class is private and not meant to be imported
    by the end user. It should only be called by an
    InequalityFeatureSet object instance to perform
    inequality decomposition of a feature.
    '''
    
    def __ge_index(self, values: Iterable, alpha: float) -> float:
        '''
        This method calculates the generalized entropy
        index, as mentioned in the KDD published paper 
        (https://arxiv.org/abs/1807.00787).
        '''
        
        mean = np.mean(values)
        if mean == 0:
            warnings.warn("0 benefit mean in GE_2 computation")
            return 0
        
        # Handling special cases for GE
        ge = None
        if alpha == 0:
            warnings.warn("Alpha parameter is set to 0, falling into special case of general entropy calculation")
            ge = np.sum(np.log(values / mean))
            ge /= -len(values)
        elif alpha == 1:
            warnings.warn("Alpha parameter is set to 1, falling into special case of general entropy calculation")
            ge = np.sum((values / mean) * np.log(values / mean))
            ge /= len(values)
        else:
            ge = np.sum(np.power(values / mean, alpha) - 1)
            ge /= (len(values) * alpha * (alpha - 1))
        return ge

    def run(self, group_benefits: Dict[str, Iterable], alpha: float = 2.0) -> Dict:
        '''
        This method can be called to calculate and return
        the decomposition for a given feature.
        
        Args:
        group_benefits (Dict[str, Iterable]): Dictionary
            in the form {<attr>:[x1, ..., xn]}, wehere an
            attribute represent a group's label and its
            values represent each of its samples' benefit
        alpha (float): Hyperparameter for calculating the
            inequality decomposition for a given group
            (by default will have a value of 2)
            
        Returns:
        Dict: Dictionary containing all the necessary data
            regarding the decomposition process (both the 
            overall indices and group-wise calculations, 
            such as 'overall inequality', 'between-group
            inequality', 'within-group inequality' and
            'unweighted intermediate components')
        '''
        overall_benefits = np.concatenate(list(group_benefits.values()))
        overall_ineq = self.__ge_index(overall_benefits, alpha)
        mean_util = np.mean(overall_benefits)

        sub_inequalities = {}
        unweighted_inequalities = {}
        intergroup_components = {}
        intergroup_inequality = 0
        for group, g_benefits in group_benefits.items():
            sub_mean_util = np.mean(g_benefits)
            means_sqr = np.power(sub_mean_util / mean_util, alpha)
            sub_inequality_weight = (len(g_benefits) / len(overall_benefits)) * means_sqr

            sub_inequality = self.__ge_index(g_benefits, alpha)
            sub_inequalities[group] = sub_inequality_weight * sub_inequality
            unweighted_inequalities[group] = sub_inequality

            intergroup_component = len(g_benefits) / (len(overall_benefits) * alpha * (alpha - 1)) * \
                    (means_sqr - 1)
            intergroup_inequality += intergroup_component
            intergroup_components[group] = intergroup_component

        decomposition_sum = sum(sub_inequalities.values()) + intergroup_inequality
        assert math.isclose(overall_ineq, decomposition_sum, rel_tol=0.1), "overall inequality {} and decomposition sum {} not close".format(overall_ineq, decomposition_sum)

        return {
            'overall_inequality': overall_ineq,
            'between_group_inequality': intergroup_inequality,
            'within_group_inequalities': sub_inequalities,
            'unweighted_inequalities': unweighted_inequalities,
            'intergroup_components': intergroup_components
        }