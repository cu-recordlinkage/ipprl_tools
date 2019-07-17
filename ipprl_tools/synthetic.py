import numpy as np
from scipy.sparse import dok_matrix


def drop_per_column(data, indicators, columns = None, drop_num = None, drop_pct = None):
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

    col_map = {k:v for k,v in zip(cols_to_drop,[data.columns.get_loc(x) for x in cols_to_drop])}
    data_len = len(data)
    if drop_num is None and drop_pct is None:
        raise Exception("Both drop_num and drop_pct are None. Please specify one or the other.")

    if drop_num is None:
        if type(drop_pct) == float:
            drop_num = np.repeat(int(drop_pct * data_len),len(cols_to_drop))
        else:
            drop_num = [int(d*data_len) for d in drop_pct]
    else:
        if type(drop_num) == int:
            drop_num = np.repeat(drop_num,len(cols_to_drop))

        
    for dnum, col in zip(drop_num, cols_to_drop):
        # Choose some indices of values to drop.
        drop_vals = np.random.choice(data_len, dnum, replace=False)
        data[col].iloc[drop_vals] = ""

        result = np.repeat({"drop":True},data_len)

        _update_indicators(indicators, col_map[col], drop_vals, "drop_per_column",result)


def string_delete(data, indicators, delete_num, delete_freq, columns = None):
    """Randomly delete characters from strings to simulate typographic errors.
    
    Arguments:
        data {DataFrame} -- A DataFrame holding the data in columnar format.
    
    Keyword Arguments:
        columns {list} -- List of columns to perform string deletion on. If None (default), run on all columns. (default: {None})
        delete_num {list or int} -- Delete between 0 and delete_num characters from a string.
        delete_freq {list or float} -- Specifies to row frequency of deletion. (delete_freq=0.2 means choose 20% of the rows to have a string deletion.)
    """

    string_apply(data,indicators,delete_num,delete_freq,_delete_func, "string_delete", columns=columns)

def string_apply(data, indicators, apply_num, apply_freq, func, name, columns=None,):
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
    col_map = {k:v for k,v in zip(columns_to_use,[data.columns.get_loc(x) for x in columns_to_use])}
    data_len = len(data)

    if type(apply_num) == list and type(apply_freq) != list or type(apply_freq) == list and type(apply_num) != list:
        raise Exception("Error: Both apply_num and apply_freq must be the same type (list or scalar).")

    if type(apply_num) != list:
        apply_num = np.repeat(apply_num,len(columns_to_use))
        apply_freq = np.repeat(apply_freq,len(columns_to_use))

    for c,anum,afreq in zip(columns_to_use,apply_num,apply_freq):
        apply_result = np.stack(data[c].apply(func,args=(anum,afreq)))
        #mod_data = apply_result[:,0]
        #indicators = apply_result[:,1]
        data[c] = apply_result[:,0]
        idcs = np.squeeze(np.argwhere(apply_result[:,1] != None))
        # result_dict = [{"apply_freq"}]
        _update_indicators(indicators,col_map[c],idcs,name,apply_result[idcs,1])

def string_transpose(data, indicators, trans_num, trans_freq, columns = None):
    """Randomly transpose characters from strings to simulate typographic errors.
    
    Arguments:
        data {DataFrame} -- A DataFrame holding the data in columnar format.
        trans_num {list or int} -- Perform this many transpositions on each string.
        trans_freq {list or float} -- Transpositions should occur this frequently.
    
    Keyword Arguments:
        columns {list or None} -- list of columns to apply the transpositions (default: {None})
    """
    string_apply(data,indicators,trans_num,trans_freq,_transpose_func,"string_transpose",columns=columns)

def string_insert_alpha(data, indicators, insrt_num, insrt_freq, columns=None):
    """Randomly insert characters from strings to simulate typographic errors.
    
    Arguments:
        data {DataFrame} -- A DataFrame holding the data in columnar format.
        insrt_num {list or int} -- Insert between 0 and num_insrt characters to a string.
        insrt_freq {list or float} -- Specifies to row frequency of insertion. (insrt_freq=0.2 means choose 20% of the rows to have a string insertion.)
    
    Keyword Arguments:
        columns {list} -- List of columns to perform string insertion on. If None (default), run on all columns. (default: {None})
    """
    string_apply(data,indicators,insrt_num,insrt_freq,_insert_func_alpha,"string_insert_alpha",columns=columns)

def string_insert_numeric(data, indicators, insrt_num, insrt_freq, columns=None):
    """Randomly insert characters from strings to simulate typographic errors.
    
    Arguments:
        data {DataFrame} -- A DataFrame holding the data in columnar format.
        insrt_num {list or int} -- Insert between 0 and num_insrt characters to a string.
        insrt_freq {list or float} -- Specifies to row frequency of insertion. (insrt_freq=0.2 means choose 20% of the rows to have a string insertion.)
    
    Keyword Arguments:
        columns {list} -- List of columns to perform string insertion on. If None (default), run on all columns. (default: {None})
    """
    string_apply(data,indicators,insrt_num,insrt_freq,_insert_func_numeric,"string_insert_numeric",columns=columns)

def soundex_string_corrupt(data, indicators, corrupt_name_pct, columns= None):
    """Function to replace elements in a columns with Soundex equivalents
    
    Arguments:
        data {DataFrame} -- A DataFrame holding the data in columnar format.
    
    Keyword Arguments:
        columns {list} -- List of columns to operate on. If None, operate on all columns (default: {None})
        corrupt_char_pct {float or list} -- Percentage of replacements to perform per column (default: {None})
    """
    try:
        from fuzzy import Soundex
    except ImportError:
        print("Error: Must install the module 'fuzzy' before running this function.")
        return

    soundex_obj = Soundex(4)

    columns_to_use = data.columns if columns is None else columns
    col_map = {k:v for k,v in zip(columns_to_use,[data.columns.get_loc(x) for x in columns_to_use])}

    data_len = len(data)

    if type(corrupt_name_pct) is not list:
        corrupt_name_pct = np.repeat(corrupt_name_pct,len(columns_to_use))

    for column,pct in zip(columns_to_use,corrupt_name_pct):
        # Generate a Soundex lookup table for this column
        lookup_table = _build_lookup_table(data[column],soundex_obj)
        # Now randomly select the indices to replace
        indcs = np.random.choice(data_len,int(pct*data_len),replace=False)

        result = np.repeat({"replaced": True},indcs.shape[0])

        # For each index, choose a replacement soundex value randomly from the set
        # of all values that have this soundex value.
        _soundex_replace(data[column].iloc[indcs],lookup_table,soundex_obj)
        _update_indicators(indicators,col_map[column],indcs,"soundex_corrupt",result)

def keyboard_string_corrupt(data, indicators, columns = None, corrupt_char_num = None, corrupt_char_pct = None):
    pass

def edit_values(data, swap_set, indicators, pct_edits, columns=None):
    # Get the columns to use for the edit values operation.
    columns_to_use = data.columns if columns is None else columns

    col_map = {k:v for k,v in zip(columns_to_use,[data.columns.get_loc(x) for x in columns_to_use])}

    data_len = len(data)

    if type(pct_edits) is not list:
        pct_edits = np.repeat(pct_edits,data_len)

    for col,ed_pct in zip(columns_to_use,pct_edits):
        # Randomly choose indices to edit
        idcs = np.random.choice(data_len,int(ed_pct*data_len),replace=False)

        swap_vals = np.random.choice(swap_set[col],int(ed_pct*data_len))

        result = np.repeat({"value_edit" : True},idcs.shape[0])

        _update_indicators(indicators,col_map[col],idcs,"edit_values",result)

        data[col].iloc[idcs] = swap_vals



### HELPER FUNCTIONS
def _build_lookup_table(column,sdx):
    lookup_table = {}
    # Compute the lookup table for this column
    for v in column:
        s_v = sdx(v)
        if lookup_table.get(s_v) is not None:
            lookup_table[s_v].append(v)
        else:
            lookup_table[s_v] = [v]
    
    # Convert lists to Numpy Arrays
    for k in lookup_table.keys():
        lookup_table[k] = np.array(lookup_table[k])
    
    return lookup_table

def _soundex_replace(column,lookup_table,sdx):
    replace_vals = []
    for i in range(len(column)):
        v = column.iloc[i]
        sdx_val = sdx(v)
        rplc_vals = lookup_table[sdx_val]
        val = np.random.choice(rplc_vals)
        replace_vals.append(val)
    return replace_vals

def _delete_func(value, del_num, del_freq):
    # Only run on some columns. The frequency of deletes is specified by delete_freq.
    if np.random.random() > del_freq or len(value) < 2:
        return [value,None]
    # Don't completely erase the string. At worst, leave one character behind.
    chars_to_delete = min(np.random.choice(del_num)+1,len(value)-1)

    # Choose some indices to delete from the string. Choose between 0 and max_chars_to_delete_indices
    num_to_delete = np.random.choice(chars_to_delete)
    idcs_to_delete = np.random.choice(len(value),num_to_delete,replace=False)

    result_dict = {"num_deleted":num_to_delete,
                   "idcs_deleted":idcs_to_delete}

    # Build a new string without the deleted characters
    new_str = "".join([c for i,c in enumerate(value) if ~np.isin(i,idcs_to_delete)])

    return [new_str,result_dict]

def _transpose_func(value, trns_num, trns_freq):
    # Only run on some columns
    if np.random.random() > trns_freq:
        return [value,None]
    
    # Get the maximum number of transposes to perform.
    max_num_of_trnsp = min((len(value)//2)-1, trns_num)

    # If the string is not long enough to perform a string transpose,
    # then return the string unchanged.
    if max_num_of_trnsp <= 0:
        return [value,None]
    # Choose the actual number of transposes to perform
    num_to_trns = np.random.choice(max_num_of_trnsp)+1
    idcs_to_trnsp = np.random.choice(len(value)//2,num_to_trns)*2

    result_dict = {"num_transposed":num_to_trns,
                   "idcs_transposed":idcs_to_trnsp}

    string_pos = np.arange(len(value))
       
    for swap_idx in idcs_to_trnsp:
        string_pos[swap_idx:swap_idx+2] = np.flip(string_pos[swap_idx:swap_idx+2])

    return ["".join([value[s] for s in string_pos]),result_dict]


def _swap(arr,l,r):
    arr[[l,r]] = arr[[r,l]]

def _get_rand_char():
    return chr(np.random.randint(97,123))

def _get_rand_num_char():
    return chr(np.random.randint(48,58))

def _insert_func_alpha(value, ins_num, ins_freq):
    # Only run on some columns
    if np.random.random() > ins_freq:
        return [value,None]
    
    # Choose the number of insertions to perform
    num_to_ins = min(np.random.choice(ins_num)+1,len(value))
    idcs_to_ins = np.random.choice(len(value),num_to_ins,replace=False)

    result_dict = {"num_inserted":num_to_ins,
                   "idcs_inserted":idcs_to_ins}

    new_str = "".join([c if ~np.isin(i,idcs_to_ins) else c+_get_rand_char() for i,c in enumerate(value)])

    return [new_str,result_dict]

def _insert_func_numeric(value, ins_num, ins_freq):
    # Only run on some columns
    if np.random.random() > ins_freq:
        return [value,None]
    
    # Choose the number of insertions to perform
    num_to_ins = min(np.random.choice(ins_num)+1,len(value))
    idcs_to_ins = np.random.choice(len(value),num_to_ins,replace=False)

    result_dict = {"num_inserted":num_to_ins,
                   "idcs_inserted":idcs_to_ins}

    new_str = "".join([c if ~np.isin(i,idcs_to_ins) else c+_get_rand_num_char() for i,c in enumerate(value)])

    return [new_str,result_dict]

def _update_indicators(indicator_df, column_index, indices, method_name, metric_result):

    tuple_pack = np.stack(np.broadcast_arrays(column_index,indices)).T

    # Iterate over the pairs and set their elements
    for ind,pair in enumerate(tuple_pack):
        tuple_pair = tuple(pair)
        if indicator_df.get(tuple_pair) is None:
            indicator_df[tuple_pair] = {}

        indicator_df[tuple_pair][method_name] = metric_result[ind]

