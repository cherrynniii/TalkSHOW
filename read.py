import pickle
from pprint import pprint

with open("214428-00_00_58-00_01_08.pkl", "rb") as fr:
    data = pickle.load(fr)
print(data.keys())

with open("pkl_contents.txt", "w", encoding="utf-8") as fw:
        pprint(data, stream=fw)