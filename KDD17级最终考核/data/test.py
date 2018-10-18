from kdd_final_test_decisionTree import *
from create_submit_file import create_submit_file
import pandas as pd
import numpy as np

data_set = np.array(pd.read_csv("test.csv"))
root = load_a_decision_tree("decision_tree_mode.pkl")
predict_data = predict(data_set, root)
print(predict_data)
create_submit_file(predict_data)


