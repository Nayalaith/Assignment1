from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load the pre-trained model and the dataset
df = pd.read_hdf('models/Netflix_embedding.h5', key='df')  # Path to precomputed embeddings
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

class MovieInput(BaseModel):
    genres: str
    emotions: str
    length: str

def get_embedding(text):
    if isinstance(text, list):
        embeddings = []
        for word in text:
            encoded_input = tokenizer(word, return_tensors='pt')
            with torch.no_grad():
                output = model(
                    input_ids=encoded_input["input_ids"],
                    attention_mask=encoded_input["attention_mask"],
                    return_dict=True
                )
            word_embedding = output.last_hidden_state[:, 0, :]
            embeddings.append(word_embedding)

        mean_emb = torch.mean(torch.stack(embeddings), dim=0)
        return mean_emb
    else:
        encoded_input = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            output = model(
                input_ids=encoded_input["input_ids"],
                attention_mask=encoded_input["attention_mask"],
                return_dict=True
            )
        mean_emb = output.last_hidden_state.mean(dim=1)
        return mean_emb

def recommend_movies(df, input_list, top_n,number_of_users=1):
    weights = [0.7, 0.1, 0.2]
    input_embedding= torch.zeros_like(torch.tensor(df['combined_embedding'].iloc[0]))
    for input in input_list:
      embedding=torch.zeros_like(torch.tensor(df['combined_embedding'].iloc[0]))
      i=0
      for key,value in input.items():
        if key in ['genres','emotions','length']:
          embedding+=get_embedding(value)*weights[i]
        i+=1
      input_embedding+=embedding
    input_embedding/=number_of_users
    df['cosine_similarity'] = df['combined_embedding'].apply(lambda x: cosine_similarity(x.reshape(1, -1), input_embedding)[0][0])
    input=input_list[0]
    top_50_movies = df.nlargest(50, 'cosine_similarity')
    #filtered_df= filter_df(top_50_movies,input['type'],input['age_certification'],input['release_year'])
    #filtered_df=filtered_df.nlargest(min(10,len(filtered_df)), 'cosine_similarity')
    top_50_movies_sorted = top_50_movies.sort_values(by='imdb_score', ascending=False)
    final_top_movies = top_50_movies_sorted.head(top_n)
    return final_top_movies['title'].tolist()

@app.post("/recommend/")
def recommend(movie_input: MovieInput):
    user_input = [{'genres': movie_input.genres, 'emotions': movie_input.emotions, 'length': movie_input.length}]
    recommendations = recommend_movies(df,user_input,top_n=5)
    return {"recommendations": recommendations}


