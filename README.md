# ipprl_tools

`ipprl_tools` is a Python package developed by the CU Record Linkage team as part of our research into Incremental Privacy Preserving Record Linkage (iPPRL). 

This package contains utility methods for generating synthetic data for record linkage, as well as functions to calculate common 'linkability' metrics, which allow a user to assess the utility of their data fields for record linkage tasks.

## Installation

The package can installed using `pip`, or used without installation by writing cloning this repository and importing `ipprl_tools` locally.

### Local Usage
To use `ipprl_tools` locally:

1. Clone this repository and create a virtual environment in the repository root directory (the directory containing this `README.md` file) using `python3 -m venv venv`.
2. Activate your virtual environment using `source venv/bin/activate` on Unix, or `venv/Scripts/activate.ps1` on Windows.
3. With the virtual environment active, install dependencies using `pip install -r requirements.txt`.
4. Create a script or notebook in the repository root directory which imports `ipprl_tools`.

You can test functionality using the `demo.ipynb` Jupyter Notebook (Jupyter should already be installed as a dependency from `requirements.txt`) or `demo.py` script, which will download some data, and calculate metrics.

### Installation via `pip`
To install the package with `pip`, run:

`pip install git+https://github.com/cu-recordlinkage/ipprl_tools`

## Usage 

## Synthetic Data Corruption

The synthetic data tools can be found at:
`ipprl_tools.synthetic.py`
and imported into your python code using:
`from ipprl_tools import synthetic`

We recommend viewing or running the tutorial Jupyter Notebook, which can be found at:
`ipprl_tools/docs/ipprl_tools Tutorial Notebook.ipynb`

## Linkability Metrics

If you would just like to run the linkability metrics on your own data, you can use the `run_metrics.py` script:

`python3 run_metrics.py --input_file=<your CSV data> --output_file=<output file path>`

to calculate all available metrics and write the result to a file.

This script uses the `metrics.get_metrics()` function to compute all metrics. If you would like finer control, the documentation `ipprl_tools/docs/IPPRL_Tools_Documentation__Synthetic_Data_Corruption_Tools.pdf` provides function names of
the metric functions, which may be called individually.

## Documentation
Documentation PDF files are available at:
`ipprl_tools/docs/IPPRL_tools_Documentation_*.pdf`.

## Issues, Questions, or Suggestions
If you encounter issues with the module or have a suggestion for improvement,
please open an issue here or email me at:

[andrew.2.hill@cuanschutz.edu](mailto:andrew.2.hill@cuanschutz.edu)
