import argparse
import pandas as pd
from ipprl_tools import metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, help="The input file (in CSV format) to calculate metrics for.")
    parser.add_argument("--output_file", required=True, default="computed_metrics.csv")

    args = parser.parse_args()

    # Read CSV into dataframe, parse all columns as string.
    data = pd.read_csv(args.input_file, dtype=str, keep_default_na=False)

    # Compute metrics on the dataframe.
    result_metrics = metrics.run_metrics(data)

    # Write metrics out to file.
    result_metrics.to_csv(args.output_file)
    print(f"Results written to '{args.output_file}'.")
