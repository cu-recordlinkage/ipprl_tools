import numpy as np
import pandas as pd

def convert_data(data):
    """Converts a DataFrame to be ready for usage with the linkage metrics."""
    str_data = data.copy().astype(np.str)
    str_data[str_data == "nan"] = ""
    
    return str_data

def run_metrics(data):
    """Runs all available metrics on a pandas DataFrame containing the data.
    Arguments:
        data {DataFrame} -- Pandas DataFrame containing the data in columnar format.
        
    Note: Data must be in a DataFrame of type np.str. Missing values should be represented by the empty string "".
    """

    # Group Size Metrics
    mean_gs = pd.Series(agg_group_size(data, agg_func=np.mean))
    std_gs = pd.Series(agg_group_size(data, agg_func=np.std))
    min_gs = pd.Series(agg_group_size(data, agg_func=np.min))
    max_gs = pd.Series(agg_group_size(data, agg_func=np.max))
    median_gs = pd.Series(agg_group_size(data, agg_func=np.median))

    # Others
    mdr = pd.Series(missing_data_ratio(data))
    dvr = pd.Series(distinct_values_ratio(data))
    se = pd.Series(shannon_entropy(data))
    tme = pd.Series(theoretical_maximum_entropy(data))
    ptme = pd.Series(percent_theoretical_maximum_entropy(data))
    atf = pd.Series(average_token_frequency(data))

    out_df = pd.concat([mean_gs, median_gs, std_gs, min_gs, max_gs, mdr, dvr, se, tme, ptme, atf], axis=1)
    out_df.columns = ["Mean Group Size", "Median Group Size", "Stdev Group Size", "Min Group Size", "Max Group Size",
                      "Missing Data Ratio", "Distinct Values Ratio", "Shannon Entropy", "Theoretical Max Entropy",
                      "% Theoretical Max Entropy", "Average Token Frequency"]
    return out_df

def missing_data_ratio(data, columns=None):
    """Computes the missing data ratio (MDR) for a given dataset.
    
    Arguments:
        data {DataFrame} -- Pandas DataFrame containing the data in columnar format
    
    Keyword Arguments:
        columns {list} -- List of columns to compute MDR for. If this is None, compute
                          for all columns by default. (default: {None})
    """ 

    column_list = data.columns if columns is None else columns

    data_len = len(data)

    return {c:np.sum(data[c] == "")/data_len for c in column_list}

def distinct_values_ratio(data, columns=None):
    """Computes the Distinct Values Ratio for a given dataset.
    
    Arguments:
        data {DataFrame} -- Pandas DataFrame containing the data in columnar format
    
    Keyword Arguments:
        columns {list} -- List of columns to compute MDR for. If this is None, compute
                          for all columns by default. (default: {None})
    """ 
    column_list = data.columns if columns is None else columns

    data_len = len(data)

    return {c:len(data[c].unique())/data_len for c in column_list}


def group_size(data, columns=None):
    """group_size
    
    Arguments:
        data {DataFrame} -- Pandas DataFrame containing the data in columnar format
        columns {list} -- List of columns to compute metrics for. If this is None, compute for all columns by default.

    Returns:
        group_sizes -- Dictionary of Counter objects in the format {column_name : Counter}
    """
    # Import the counter library
    from collections import Counter
    
    # Get the list of columns to calculate group size for.
    column_list = data.columns if columns is None else columns

    # Get the group sizes
    group_sizes = {col:Counter(data[col]) for col in column_list}

    for val in group_sizes.values():
        # Remove NaN (missing counts)
        if val.get("") is not None:
            del val[""]
            
    # Return the group sizes.
    return group_sizes


def agg_group_size(data, agg_func = np.mean, columns=None):
    """agg_group_size
    
    Arguments:
        data {DataFrame} -- DataFrame containing the data in columnar format
    
    Keyword Arguments:
        agg_func {function} -- Function to perform aggregation on the data. Should accept a list of values,
                               and return a single aggregate value. (default: {np.mean})
        columns {list} -- List of columns to perform aggregation on (default: {None})
    
    Returns:
        Dictionary of aggregated values in the form {column_name : aggregate_value}
    """
    group_sizes = group_size(data,columns=columns)
    
    columns_to_use = data.columns if columns is None else columns  

    agg_group_sizes = []
    for col in group_sizes:
        if len(group_sizes[col]) == 0:
            agg_group_sizes.append(-1)
        else:
            agg_group_sizes.append(agg_func([*group_sizes[col].values()]))
    #agg_group_sizes = [agg_func([*group_sizes[col].values()]) for col in group_sizes]

    return {col:agg_val for col,agg_val in zip(columns_to_use,agg_group_sizes)}

def shannon_entropy(data, columns=None):
    """Function to compute the Shannon Entropy for a set of columns.
    
    Arguments:
        data {DataFrame} -- DataFrame containing the data in columnar format
    
    Keyword Arguments:
        columns {list} -- List of columns to perform aggregation on. Default computes for all columns. (default: {None})
    """
    from collections import Counter
    from scipy.stats import entropy

    column_list = data.columns if columns is None else columns

    # Get counts of each value for each column
    count_vals = [Counter(data[c].values) for c in column_list]

    # Probability for a single value in a discrete distribution is (num_of_occurrences/len_of_distribution)
    probabilities = np.array([np.array(list(count.values()))/len(count) for count in count_vals])
    entropies = {col:entropy(p,base=2) for col,p in zip(column_list,probabilities)}
    return entropies

def joint_entropy(data, columns=None):
    """Function to compute the Joint Entropy between two columns.

    Arguments:
        data {DataFrame} -- DataFrame containing the data in columnar format.
    
    Keyword Arguments:
        columns {list of tuple} -- List of tuples. Each tuple represents a pair of columns to compute the joint entropy for.
    """
    pass
    
def theoretical_maximum_entropy(data, columns=None):
    """Calculates the Theoretical Maximum Entropy for a given dataset.
    
    Arguments:
        data {DataFrame} -- DataFrame containing the data in columnar format.
    
    Keyword Arguments:
        columns {list} -- List of columns to calculate the TME for. If None (default), calculate for all
                          columns.  (default: {None})
    
    Returns:
        Dictionary of {column_name : TME(column)}
    """
    columns_to_use = data.columns if columns is None else columns

    unq_vals = [len(set(data[c])) for c in columns_to_use]

    tmes = [-np.log2(1/uq) if uq is not 0 else 0 for uq in unq_vals]

    return {c:tme for c,tme in zip(columns_to_use,tmes)}

def percent_theoretical_maximum_entropy(data, columns=None):
    """Calculates the Percentage of Theoretical Maximum Entropy for a given dataset.
    
    Arguments:
        data {DataFrame} -- DataFrame containing the data in columnar format.
    
    Keyword Arguments:
        columns {list} -- Columns to calculate the P_TME for. If None (default), calculate for all
                          columns. (default: {None})
    
    Returns:
        p_tme -- Dictionary of {column_name : P_TME(column)}
    """
    tme = theoretical_maximum_entropy(data=data,columns=columns)
    entropy = shannon_entropy(data=data,columns=columns)
    # Calculate the P_TME for each column in the column list.
    p_tme = {}
    for k in tme.keys():
        if tme[k] == 0:
            pct = np.nan
        else:
            pct = entropy[k]/tme[k] * 100
        p_tme[k] = pct
    return p_tme


def average_token_frequency(data, columns=None):
    """Calculate the Average Token Frequency for a given dataset.
    
    Arguments:
        data {DataFrame} -- DataFrame containing the data in columnar format. 
    
    Keyword Arguments:
        columns {list} -- List of columns to calculate the ATF for. If None (default), calculate for all
                          columns. (default: {None})
    
    Returns:
        atfs -- Dictionary of {column_name : ATFS(column)}
    """
    columns_to_use = data.columns if columns is None else columns
    # Calculate the ATF for each column in the column list.
    atfs = {}
    data_len = len(data)


    for c in columns_to_use:
        # TODO: Should we count missing values in the numerator or denominator of these metrics?
        data_set = set(data[c])
        if "" in data_set:
            data_set.remove("")
        atfs[c] = data_len/len(data_set) if len(data_set) > 0 else np.nan

    return atfs


    

    
    
