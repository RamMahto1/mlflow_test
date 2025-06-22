# main.py

from utils.data_loader import load_data
from experiments.decision_tree import run

X_train, X_test, y_train, y_test = load_data()

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)



if __name__=="__main__":
    run()
