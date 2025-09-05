import json
import pickle
__locations = None
__data_columns = None
__model = None


def get_location_names():
    return __locations

def load_saved_artifacts():
    print("Loading saved artifacts..")
    global __data_columns
    global __locations
    with open("W:/vscode/Machine-Learning/MLPROJECT/server/artifacts/columns,json","r") as f:
        __data_columns = json.load(f)["data_columns"]
        __locations = __data_columns[3:]

    with open("W:/vscode/Machine-Learning/MLPROJECT/server/artifacts/home_price.pickle", "rb") as f:
        __model = pickle.load(f)
    print("Done loading everything")

if __name__ == "__main__":
    load_saved_artifacts()
    print(get_location_names())