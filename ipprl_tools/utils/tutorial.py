DATA_FILE_PATH = "https://drive.google.com/uc?export=view&id=1b2P-SLIrcTaAc9I9xul_yDlHjo03PuMd"
DATA_DIR = "data/"
SAMPLE_DATA_FILE = "sample_data.zip"

from urllib.request import urlretrieve
from os.path import isfile, isdir, exists, join, split
from os import makedirs

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
