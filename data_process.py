import os
import re
import pandas as pd


def merge_text(text_list):
    i = 0
    j = 1

    k = len(text_list)

    while j < k:
        if len(text_list[i].split()) <= 30:
            text_list[j] = text_list[i] + " " + text_list[j]
            text_list[i] = " "
        i += 1
        j += 1

    return [accepted for accepted in text_list if accepted is not " "]


def get_text(path):
    doc_list = sorted(os.listdir(path))
    text = []
    for doc in doc_list:
        sub_text = []
        with open(os.path.join(path, doc), encoding='utf-8') as f:
            for line in f.readlines():
                temp_text = re.sub("\\n", "", line)
                if temp_text != "":
                    sub_text.append(temp_text)

            sub_text = merge_text(sub_text)
            text.extend(sub_text)
    return text


def dataframe(path):
    text = get_text(path)
    df = {
        "text": text
    }
    df = pd.DataFrame(df)
    df.to_csv("Responses.csv")
