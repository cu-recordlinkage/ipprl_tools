import numpy as np

def drop_per_column(data, columns = None, drop_num = None, drop_pct = None):
    """Function to randomly drop data values in columns 
    
    Arguments:
        data {DataFrame} -- Pandas DataFrame holding the data.
    
    Keyword Arguments:
        columns {list} -- List of columns to drop data from. (default: {None})
        drop_num {list or scalar} -- If list, specifies number of drops per column in 'columns',
                                     otherwise, specifies the same number of drops for each column
                                     in 'columns'. (default: {None})
        drop_pct {list or scalar} -- If list, specifies the percentage of data values to drop for
                                     each column in 'columns'. Otherwise, specifies the same 
                                     percentage of drops for each column in 'columns' (default: {None})
    """

    cols_to_drop = data.columns if columns is None else columns

    if drop_num is None and drop_pct is None:
        raise Exception("Both drop_num and drop_pct are None. Please specify one or the other.")
    
    if drop_num is not None:
        if type(drop_num) == list:
            for dnum,col in zip(drop_num,cols_to_drop):
                # Choose some indices of values to drop.
                drop_vals = np.random.choice(len(data),dnum,replace=False)
                data[col].iloc[drop_vals] = ""
        elif type(drop_num) == int:
            for col in cols_to_drop:
                drop_vals = np.random.choice(len(data),drop_num,replace=False)
                data[col].iloc[drop_vals] = ""
        else:
            raise ValueError(drop_num)
    else:
        if type(drop_pct) == list:
            for dpct,col in zip(drop_pct,cols_to_drop):
                # Choose some indices of values to drop.
                dnum = int(dpct*len(data))
                drop_vals = np.random.choice(len(data),dnum,replace=False)
                data[col].iloc[drop_vals] = ""
        elif type(drop_pct) == int:
            dnum = int(drop_pct*len(data))
            for col in cols_to_drop:
                drop_vals = np.random.choice(len(data),dnum,replace=False)
                data[col].iloc[drop_vals] = ""

def string_delete(data, delete_num, delete_freq, columns = None):
    """Randomly delete characters from strings to simulate typographic errors.
    
    Arguments:
        data {DataFrame} -- A DataFrame holding the data in columnar format.
    
    Keyword Arguments:
        columns {list} -- List of columns to perform string deletion on. If None (default), run on all columns. (default: {None})
        delete_num {list or int} -- Delete between 0 and delete_num characters from a string.
        delete_freq {list or float} -- Specifies to row frequency of deletion. (delete_freq=0.2 means choose 20% of the rows to have a string deletion.)
    """

    string_apply(data,delete_num,delete_freq,_delete_func,columns=columns)

def string_apply(data, apply_num, apply_freq, func, columns=None):
    """Apply a function to modify strings in a dataset.
    
    Arguments:
        data {DataFrame} -- A DataFrame holding the data in columnar format.
        apply_num {list or int} -- Apply to function to between 0 and apply_num characters in the string.
        apply_freq {list or float} -- Specifies the row frequency of function application (apply_freq=0.2 means choose 20% of the rows to apply the function.)
        func {callable} -- Callable function that modifies each string value.
    
    Keyword Arguments:
        columns {list} -- List of columns to perform string manipulation on. If None (default), run on all columns.  (default: {None})
    """
    columns_to_use = data.columns if columns is None else columns

    if type(apply_num) == list and type(apply_freq) != list or type(apply_freq) == list and type(apply_num) != list:
        raise Exception("Error: Both delete_num and delete_freq must be the same type (list or scalar).")
    
    if type(apply_num) == list:
        for c,anum,afreq in zip(columns_to_use,apply_num,apply_freq):
            data[c] = data[c].apply(func,args=(anum,afreq))
    elif type(apply_num) == int:
        for c in columns_to_use:
            data[c] = data[c].apply(func,args=(apply_num,apply_freq))


def string_transpose(data, columns = None, num_trnsp = None):
    pass

def string_insert(data, insrt_num, insrt_freq, columns=None):
    """Randomly insert characters from strings to simulate typographic errors.
    
    Arguments:
        data {DataFrame} -- A DataFrame holding the data in columnar format.
        insrt_num {list or int} -- Insert between 0 and num_insrt characters to a string.
        insrt_freq {list or float} -- Specifies to row frequency of insertion. (insrt_freq=0.2 means choose 20% of the rows to have a string insertion.)
    
    Keyword Arguments:
        columns {list} -- List of columns to perform string insertion on. If None (default), run on all columns. (default: {None})
    """
    string_apply(data,insrt_num,insrt_freq,_insert_func,columns=columns)

def soundex_string_corrupt(data, columns= None, corrupt_char_num = None, corrupt_char_pct = None):
    pass

def keyboard_string_corrupt(data, columns = None, corrupt_char_num = None, corrupt_char_pct = None):
    pass

def edit_values(data, swap_set, columns=None, num_edits = None):
    pass

### HELPER FUNCTIONS

def _delete_func(value, del_num, del_freq):
    # Only run on some columns. The frequency of deletes is specified by delete_freq.
    if np.random.random() > del_freq:
        return value
    # Don't completely erase the string. At worst, leave one character behind.
    max_chars_to_delete = min(len(value)-1, del_num)

    # Choose some indices to delete from the string. Choose between 0 and max_chars_to_delete_indices
    num_to_delete = np.random.choice(max_chars_to_delete)
    idcs_to_delete = np.random.choice(len(value),num_to_delete,replace=False)

    # Build a new string without the deleted characters
    new_str = "".join([c for i,c in enumerate(value) if ~np.isin(i,idcs_to_delete)])

    return new_str

def _transpose_func(value, trns_num, trns_freq):
    # Only run on some columns
    if np.random.random() > trns_freq:
        return value
    
    # Get the maximum number of transposes to perform.
    max_num_of_trnsp = min(len(value)-1, trns_num)

    # Choose the actual number of transposes to perform
    num_to_trns = np.random.choice(max_num_of_trnsp)
    idcs_to_trnsp = np.random.choice

def _swap(arr,l,r):
    arr[[l,r]] = arr[[r,l]]

def _get_rand_char():
    return chr(np.random.randint(97,123))

def _insert_func(value, ins_num, ins_freq):
    # Only run on some columns
    if np.random.random() > ins_freq:
        return value
    
    # Choose the number of insertions to perform
    num_to_ins = np.random.choice(ins_num)
    idcs_to_ins = np.random.choice(len(value),num_to_ins,replace=False)

    new_str = "".join([c if ~np.isin(i,idcs_to_ins) else c+_get_rand_char() for i,c in enumerate(value)])
    return new_str