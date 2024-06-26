from ipprl_tools import metrics
from ipprl_tools.utils.data import get_data
import pandas as pd

if __name__ == "__main__":
    # This function will download some demo data, and provide the file path.
    print("Loading data...")
    demo_data_path = get_data()
    # We can read in the demo data from above.
    # Note: In order for the metrics calculation to work correctly, it is important that we call the read_csv function with the following arguments:
    # dtype=str - Enforces that all columns are read in as strings (as opposed to numeric/date/etc). This is important as the metrics calculations assume they are working with strings.
    # keep_default_na=False - Enforces that missing values are coded as '' instead of np.NaN or some other value. The metric calculations will also assume that empty string '' is the missing value placeholder.
    demo_data = pd.read_csv(demo_data_path, dtype=str, keep_default_na=False)
    # Calculate the metrics for the demo data.
    print("Data loaded, starting metric calculation...")
    result_metrics = metrics.run_metrics(demo_data)
    # Write the metrics to CSV file.
    out_path = "demo_metrics_out.csv"
    result_metrics.to_csv(out_path, index_label="data_column")
    print(f"Metrics have been written to '{out_path}'.")