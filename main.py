from math  import sqrt
import re
d = sqrt(2)
print(2*d)
 
# Data clean and preprocessing

## Get the data

import pandas as pd
df = pd.read_csv('data_set/1429_1.csv',low_memory=False)

def preprocess_data(text):
    # make lower case
    text = text.lower()
    # remove punctuation
    text = text.replace('.', '').replace(',', '').replace('!', '').replace('?', '')
    return text



# Apply the preprocessing function to an example and see before and after
example_text = "Hello, World! This is a test."
print("Before preprocessing:", example_text)
print("After preprocessing:", preprocess_data(example_text))


# Input features from the data
df_input = df.loc[:, ['name', 'categories', 'reviews.rating', 'reviews.text']]

print(df_input.head())

# preprocess the input features
df_input['name'] = df_input['name'].str.strip()
df_input['categories'] = df_input['categories'].str.lower().str.strip()
df_input['reviews.text'] = df_input['reviews.text'].str.lower().str.strip()
df_input['reviews.rating'] = df_input['reviews.rating'].astype(str)

# Remove non-ASCII characters from review_body and review_title

def remove_non_ascii(text):
    return None



# Remove rows with NaN values in the specified columns
df_input = df_input.dropna(subset=['name', 'categories', 'reviews.rating', 'reviews.text'])

 # Remove html tags from reviews
def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

df_input['reviews.text'] = df_input['reviews.text'].apply(remove_html_tags)

# Remove extra whitespace from reviews
def remove_extra_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()

df_input['reviews.text'] = df_input['reviews.text'].apply(remove_extra_whitespace)

# Remove leading and trailing whitespace from product names and categories
df_input['name'] = df_input['name'].str.strip()
df_input['categories'] = df_input['categories'].str.strip() 

# review columns
print(df_input[['reviews.text', 'reviews.rating']].head(3))

# Create a new column for the training data
df_input['prompt'] = 'Product: ' + df_input['name'] + '\nCategory: ' + df_input['categories'] + '\nRating: ' + df_input['reviews.rating'] 

print('\n',df_input['prompt'][1])


##################################

# split data into training and target and combinate them in one column
df_input['training_data'] = df_input['prompt'] + ' ' + df_input['reviews.text']


# Load the training data into JSON to be used later in Hugging Face
import json

with open("training_data.jsonl", "w") as f:
    for _, row in df_input.iterrows():
        json.dump({"text": row['training_data']}, f)
        f.write("\n")

from datasets import load_dataset

dataset = load_dataset('json', data_files='training_data.jsonl', split='train')

