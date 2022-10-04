from data_process import *
from retrieval import *
import argparse


parser = argparse.ArgumentParser(description="Run the query for the bot")
parser.add_argument('--query', help="Question to the bot", type=str, required=True)
parser.add_argument('--data_path', help="Path for the stored dataset", type=str, required=True)

args = parser.parse_args()
path = args.data_path
query = args.query

if "Responses.csv" not in os.listdir(os.getcwd()):
    dataframe(path)

df = pd.read_csv("Responses.csv")
text = list(df["text"].values)


search(query, text)

