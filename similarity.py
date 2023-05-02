import numpy as np
import openai
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

df = pd.read_csv('final_qa.csv')

COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"


def get_embedding(text: str, model: str=EMBEDDING_MODEL) -> list:
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def load_embeddings(fname: str) -> dict:
    """
    Read the document embeddings and their keys from a CSV.
    
    fname is the path to a CSV with exactly these named columns: 
        "title", "heading", "0", "1", ... up to the length of the embedding vectors.
    """
    
    df = pd.read_csv(fname, header=0)
    max_dim = max([int(c) for c in df.columns if c != "Title" and c != "Heading"])
    return {
           (r.Title, r.Heading): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()}




def vector_similarity(x: list, y: list) -> float:
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, contexts: dict) -> list:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """

    similar_doc_list = []
    candidate = []
    query_embedding = get_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)

    for doc in document_similarities:
      similar_doc_list.append(doc[1])

    for items in similar_doc_list:
      title = items[0]
      heading = items[1]
      row = df[(df['Title'] == title) & (df['Heading'] == heading)]
      context_arr = row["Context"].values
      # print(context)
      candidate.append(context_arr[0])

    len_cand = len(candidate)

    if len_cand<=2:
      return candidate
    else:
      return candidate[:15]
    
def get_similar_context(query):
  document_embeddings = load_embeddings("final_embeddings.csv")
  context_list = (order_document_sections_by_query_similarity(query, document_embeddings))
  context = " ".join(context_list)
  
  return context.replace('\n',' ')



