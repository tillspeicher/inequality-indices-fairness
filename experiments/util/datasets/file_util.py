import os
import gzip
import numpy as np
from random import seed, shuffle
import urllib.request, urllib.error, urllib.parse
from sklearn import preprocessing
import sys
sys.path.insert(0, '../')
import output

SEED = 1122334455
seed(SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(SEED)


def load_data(data_input, attr_map, load_data_size=None):

    """
        if load_data_size is set to None (or if no argument is provided), then we load and return the whole data
        if it is a number, say 10000, then we will return randomly selected 10K examples
    """

    attrs = attr_map['attrs']
    cont_attrs = attr_map['cont_attrs']
    sensitive_attrs = attr_map['sens_attrs']
    rel_sens_vals = attr_map['rel_sens_vals']
    no_hot_encode_attrs = set(attr_map['no_hot_encode_attrs'])

    X = []
    y = []
    x_control = {}

    attrs_to_vals = {} # will store the values for each attribute for all users
    for k in attrs:
        if k in sensitive_attrs:
            x_control[k] = []
        else:
            attrs_to_vals[k] = []

    for line, class_label in data_input:
        y.append(class_label)

        for attr_val, attr_name in zip(line, attrs):
            if attr_name in sensitive_attrs:
                x_control[attr_name].append(attr_val)
            else:
                attrs_to_vals[attr_name].append(attr_val)

    def convert_attrs_to_ints(d): # discretize the string attributes
        for attr_name, attr_vals in d.items():
            if attr_name in cont_attrs:
                d[attr_name] = preprocessing.scale(attr_vals)
            else:
                le = preprocessing.LabelEncoder()
                le.fit(attr_vals)
                d[attr_name] = le.transform(attr_vals)
                if len(le.classes_) <= 2:
                    # no need to one-hot-encode later
                    no_hot_encode_attrs.add(attr_name)

            #    continue
            #uniq_vals = sorted(list(set(attr_vals))) # get unique values

            ## compute integer codes for the unique values
            #val_dict = {}
            #for i in range(0,len(uniq_vals)):
            #    val_dict[uniq_vals[i]] = i

            ## replace the values with their integer encoding
            #for i in range(0,len(attr_vals)):
            #    attr_vals[i] = val_dict[attr_vals[i]]
            #d[attr_name] = attr_vals

    # convert the discrete values to their integer representations
    #convert_attrs_to_ints(x_control)
    convert_attrs_to_ints(attrs_to_vals)

    def binarize_sensitive_features(control_map):
        binarized_control_map = {}
        #assert len(control_map) == 1 # just for simplicity now
        for sens_attr, sens_vals in control_map.items():
            lb = preprocessing.LabelBinarizer()
            lb.fit(sens_vals)
            sens_vals = lb.transform(sens_vals)
            #control_map.clear()
            for val_name, val_col in zip(lb.classes_, sens_vals.T):
                full_name = sens_attr + '_' + val_name
                val_col = np.array(val_col) #, dtype=bool)
                if full_name in rel_sens_vals or len(lb.classes_) <= 2:
                    binarized_control_map[full_name] = val_col
        return binarized_control_map

    if x_control:
        x_control = binarize_sensitive_features(x_control)
   
    # only keep people belonging to certain sensitive groups
    # if there are no senstive groups, select everyone
    idx = np.zeros(len(y), dtype=bool) if x_control else \
            np.ones(len(y), dtype=bool)
    for k, v in x_control.items():
        if k in rel_sens_vals:
            print('{}: {} people'.format(k, sum(v)))
            idx = np.logical_or(idx, v)

    attr_names = []

    # if the integer vals are not binary, we need to get one-hot encoding for them
    for attr_name in attrs:
        if attr_name in sensitive_attrs:
            continue
        attr_vals = attrs_to_vals[attr_name]
        if attr_name in cont_attrs or attr_name in no_hot_encode_attrs:
            X.append(attr_vals)
            attr_names.append(attr_name)

        else:
            #avs, index_dict = get_one_hot_encoding(attr_vals)
            ohe = preprocessing.OneHotEncoder(sparse=False)
            attr_vals = attr_vals.reshape(len(attr_vals), 1)
            attr_vals = ohe.fit_transform(attr_vals)
            for i, inner_col in enumerate(attr_vals.T):
                X.append(inner_col) 
                attr_names.append(attr_name + '_' + str(i))

    # convert to numpy arrays for easy handling
    #print('X:', X[:,0:2])
    X = np.array(X, dtype=float).T
    y = np.array(y, dtype = float)
    #for k, v in list(x_control.items()): x_control[k] = np.array(v, dtype=float)

    X = X[idx]
    y = y[idx]
    for k, v in x_control.items():
        x_control[k] = v[idx]
        
    # shuffle the data
    perm = list(range(0,len(y))) # shuffle the data before creating each fold
    shuffle(perm)
    X = X[perm]
    y = y[perm]
    for k in list(x_control.keys()):
        x_control[k] = x_control[k][perm]

    # see if we need to subsample the data
    if load_data_size is not None:
        print("Loading only %d examples from the data" % load_data_size)
        X = X[:load_data_size]
        y = y[:load_data_size]
        for k in list(x_control.keys()):
            x_control[k] = x_control[k][:load_data_size]

    print('Loaded {} people, {} from pos and {} from neg class'.format(len(y),
            np.sum(y == 1.), np.sum(y == -1.)))

    return X, y, x_control, attr_names


def get_data_rows(dataset, base_addr, data_files, separator=',',
        inner_file_reader=None):
    for f in data_files:
        data_file = check_data_file(dataset, base_addr, f)

        print('reading file', f)

        binary_input = f.endswith('.gz') or f.endswith('.zip')
        read_func = inner_file_reader if inner_file_reader is not None else gzip.open if binary_input else open
        for line in read_func(data_file, 'r'):
            if binary_input:
                line = line.decode()
            line = line.strip()
            if line == "": continue # skip empty lines
            line = [el.strip() for el in line.split(separator)]
            yield line


def check_data_file(dataset, base_url, fname):
    #files = os.listdir(".") # get the current directory listing
    files_dir = os.path.dirname(os.path.realpath(__file__)) + '/data/' + dataset # get path of this file
    output.create_dir(files_dir)
    files = os.listdir(files_dir) # get the current directory listing
    print("Looking for file '%s' in the current directory..." % fname)
    full_file = "{}/{}".format(files_dir, fname)

    if fname not in files:
        print("'%s' not found! Downloading ..." % fname)
        response = urllib.request.urlopen(base_url + fname)
        content_charset = response.info().get_content_charset()
        if content_charset is not None:
            # string file
            data = response.read().decode(response.info().get_content_charset(), 'ignore')
            write_spec = "w"
        else:
            # binary file
            data = response.read()
            write_spec = "wb"
        with open(full_file, write_spec) as fileOut:
            fileOut.write(data)
        print("'%s' download and saved locally.." % fname)
    else:
        print("File found in current directory..")

    return full_file

