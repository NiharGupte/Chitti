from data_process import *
from retrieval import *
import argparse

path = r"Data_chitti_final"

if "Responses.csv" not in os.listdir(os.getcwd()):
    dataframe(path)

df = pd.read_csv("Responses.csv")
text = list(df["text"].values)

search("what is the full form of UGAC", text)

