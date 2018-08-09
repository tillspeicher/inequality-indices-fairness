# Fairness Analysis Module

This module was created to automate the process of performing *inequality decomposition*, as described in the [paper published at KDD](https://arxiv.org/abs/1807.00787).

## Instructions

### 0. Ad-hoc

If nothing else, just take the snippet below and change it to your own taste and data. If you don't like to read, then this sample should guide you through our module but if that's not the case, five minutes is all we need to show you how it works (and you can simply skip to the next section).

'''
# Importing the only necessary class (InequalityFeatureSet)
from inequality_indices import InequalityFeatureSet

# Declaring a nested dictionary with calculated benefits
valid_dictionary = {'feature_1': {'a': [1, 3],'b': [2, 4]},'feature_2': {'a': [1, 2],'b': [3, 4]}}

# Creating a new instance of an InequalityFeatureSet
ineq_feature_set = InequalityFeatureSet(valid_dictionary)

# We can now print the decomposition
print(ineq_feature_set.get_decomposition())
'''

### 1. Introduction

To use this module you simply have to import the *InequalityFeatureSet* class:

'''
from inequality_indices import InequalityFeatureSet
'''

For it to perform the operations you want, you have to initialize it with the set of features you want to decompose. They can be provided both as a [pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html) or as a nested dictionary. As an example, we will show you two equivalent examples with these valid formats:

'''
from inequality_indices import InequalityFeatureSet

valid_dataframe = pd.DataFrame.from_dict(
    {
        'feature_1': ['a', 'b', 'a', 'b'],
        'feature_2': ['a', 'a', 'b', 'b'],
        'benefit': [1, 2, 3, 4]
    }, orient='columns'
)

valid_dictionary = {
    'feature_1': {
        'a': [1, 3],
        'b': [2, 4]
    },
    'feature_2': {
        'a': [1, 2],
        'b': [3, 4]
    }
}
'''

Notice that the main difference is how you provide the *benefit vetors*. In a [pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html) each entry has to have a pre-calculated benefit in a specific column - called *"benefit"* - whereas they can be directly fed to each features' attribute as a list when using a nested dictionary.

### 2. Decomposing Inequality

To perform the decomposition is easy now:

'''
from inequality_indices import InequalityFeatureSet

# We will use the nested dictionary in this example
valid_dictionary = {'feature_1': {'a': [1, 3],'b': [2, 4]},'feature_2': {'a': [1, 2],'b': [3, 4]}}

# You can now initialize the InequalityFeatureSet object
ineq_feature_set = InequalityFeatureSet(valid_dictionary)

# We can now print the decomposition
print(ineq_feature_set.get_decomposition())
'''

### 3. Custom Operations

As a last comment, our module allows for customizeable decomposition formatting. By passing a function as a parameter to `get_decomposition` you can even sort or rank features according to their inequality (or simply cleanup your output):

'''
from inequality_indices import InequalityFeatureSet

# We will use the nested dictionary in this example
valid_dictionary = {'feature_1': {'a': [1, 3],'b': [2, 4]},'feature_2': {'a': [1, 2],'b': [3, 4]}}

# You can now initialize the InequalityFeatureSet object
ineq_feature_set = InequalityFeatureSet(valid_dictionary)

# We can now print the decomposition
print(ineq_feature_set.get_decomposition())

# Custom method for ranking decompositions
def rank(x):
    tuples = [(key, values['between_group_inequality']) for key, values in x.items()]
    return sorted(tuples, key=lambda x: x[1], reverse=True)

# Sorted decomposition with between-group inequalities
print(ineq_feature_set.get_decomposition(rank))
'''