from kdd_final_test_decisionTree import *
import pandas as pd
import numpy as np

data_set = np.array(pd.read_csv("test.csv"))
root = load_a_decision_tree("decision_tree_mode.pkl")
predict_data = predict(data_set, root)
print(print_decision_tree(root))
print(predict_data)
print(len(predict_data))
