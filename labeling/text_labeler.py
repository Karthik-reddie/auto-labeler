import pandas as pd
from transformers import pipeline

def label_text_df(df, text_column, labels):
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    labeled_data = []
    for text in df[text_column]:
        result = classifier(text, labels)
        label = result["labels"][0]
        labeled_data.append(label)

    df["Predicted Label"] = labeled_data
    return df
