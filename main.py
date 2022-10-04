from data_process import *
from retrieval import *
import argparse

path = r"Data_chitti_final"

if "Responses.csv" not in os.listdir(os.getcwd()):
    print("Dataset doesn't exist. Creating the dataset ... ")
    dataframe(path)
    print("Dataset created")
else:
    print("Dataset already exists")

df = pd.read_csv("Responses.csv")
text = list(df["text"].values)

search("how to do two factor authentication", text)

