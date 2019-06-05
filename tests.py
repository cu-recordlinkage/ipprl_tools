import metrics
import synthetic
import pandas as pd
import numpy as np

data = pd.read_csv("data/data1.csv")
swap_set = pd.read_csv("data/data2.csv")

data = data.astype(np.str)

old_data = data.copy()


synthetic.string_transpose(data,4,0.05)
print("Transpose Complete.")
synthetic.string_delete(data,3,0.05)
print("Delete Complete.")
synthetic.string_insert(data,3,0.05)
print("Insert Complete.")
synthetic.soundex_string_corrupt(data,0.1,columns=["first_name","last_name"])
print("Soundex Corrupt Complete.")
synthetic.edit_values(data,swap_set,0.1)
print("Edit Values Complete.")
synthetic.drop_per_column(data,drop_pct=0.05)
print("Per-Column Drop Complete.")

print("Done!")