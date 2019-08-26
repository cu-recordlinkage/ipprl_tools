from urllib.request import urlretrieve
from os.path import isfile, isdir, exists, join, split
from os import makedirs
import numpy as np
import pandas as pd

DATA_FILE_PATH = "https://drive.google.com/uc?export=view&id=1b2P-SLIrcTaAc9I9xul_yDlHjo03PuMd"
DATA_DIR = "data/"
SAMPLE_DATA_FILE = "sample_data.zip"

def download_data():
	""" Download some pre-made data into the data directory """
	
	# Get the path to the module source file directory. 
	modpath = __get_source_dir()
	data_path = join(modpath,DATA_DIR)
	print("Using data path '%s'" % data_path)
	if isdir(data_path):
		if isfile(join(data_path,SAMPLE_DATA_FILE)):
			print("Sample data already exists.")
		else:
		# Download the data into the folder.
			__download_url(DATA_FILE_PATH,join(data_path,SAMPLE_DATA_FILE))
	else:
		print("Directory '%s' does not exist, creating..." % data_path) 
		# Make the directory
		makedirs(data_path)
		# Download data into folder
		__download_url(DATA_FILE_PATH,join(data_path,SAMPLE_DATA_FILE))

def __download_url(url,path):
	# Retrieve data from a URL.
	print("Downloading data from URL: %s to local file: '%s'" % (url,path))
	try:
		urlretrieve(url,path)
	except Exception as exc:
		print("ERROR:", exc)
		return
	print("Download complete. File available at: '%s'" % path)

def __get_source_dir():
	# Get the path to the module source file directory. 
	return split(split(__file__)[0])[0]

def get_data():
	# Check if data already exists.
	file_path = join(__get_source_dir(),DATA_DIR,SAMPLE_DATA_FILE)
	
	if not isfile(file_path):
		download_data()
	return file_path
	
def split_dataset(raw_data, overlap_pct=0.5):
    """
    A function to help with preparing raw data for creation of a synthetic dataset.

    This function will first randomly select a percentage of records which should be used as ground-truth "True"
    linkages. Then, the function generates two DataFrames, each containing the overl


    :param raw_data: A Pandas DataFrame containing the raw data to build the dataset from.
    :param overlap_pct:
    :return:
    """
    # Length of raw_data (number of rows)
    data_len = len(raw_data)
    # Reutrns a shuffled view of raw_data.
    shuffled = raw_data.iloc[np.random.permutation(data_len)]
    # Get the exact number of overlapping rows we should select.
    num_overlap = int(data_len * overlap_pct)
    # Get the rows that will overlap between the two datasets.
    overlap_rows = shuffled.iloc[:num_overlap]
    # We will evenly split the non-overlapping records between the two halves of the dataset, so calculate
    # How much each "half" gets here.
    unq_rows = (data_len - num_overlap) // 2

    # The first dataset gets all the overlapping rows, plus half of the non-overlapping rows
    left_df = pd.concat([overlap_rows, shuffled.iloc[num_overlap:num_overlap + unq_rows]]).copy().reset_index(drop=True)
    # How many rows will the "matching" rows be offset by? Used for calculating the ground truth pairs
    offset = len(left_df)
    # Left DataFrame has IDs from [0-offset)
    left_df["id"] = range(offset)
    left_df.set_index("id", inplace=True)
    # The second dataset gets all the overlapping rows, plus half of the non-overlapping rows.
    # If the number of leftovers isn't even, this DataFrame will have one extra row.
    right_df = pd.concat([overlap_rows, shuffled.iloc[num_overlap + unq_rows:]]).copy().reset_index(drop=True)
    # Right DataFrame has IDs from [offset-data_len)
    right_df["id"] = range(offset, offset + len(right_df))
    right_df.set_index("id", inplace=True)

    ground_truth = [(num, num + offset) for num in range(num_overlap)]

    return left_df, right_df, ground_truth

