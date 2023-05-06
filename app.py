import os
import openai
from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
load_dotenv()

app = Flask(__name__)
CORS(app)


openai.api_key =  os.getenv('API_KEY')

file_path = os.path.join(os.path.dirname(__file__))
final_qa_path = os.path.join(file_path, 'final_qa.csv')
df = pd.read_csv(final_qa_path)

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
      context_arr = row["Context"].values + row["Questions"].values + row["Answers"].values
      # print(context)
      candidate.append(context_arr[0])

    len_cand = len(candidate)

    if len_cand<=2:
      return candidate
    else:
      return candidate[:10]
    
def get_similar_context(query):
  file_path_data = os.path.join(file_path, 'final_embeddings.csv')
  document_embeddings = load_embeddings(file_path_data)
  context_list = (order_document_sections_by_query_similarity(query, document_embeddings))
  context = " ".join(context_list)
  
  return context.replace('\n',' ')

# query = input()
# context = (get_similar_context(query)).strip()
start_sequence = "\nA:"
restart_sequence = "\n\nQ: "
# prompt_input_u = f'{context}'


def func(value):
    return ''.join(value.splitlines())




def get_response(prompt_input):
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt= prompt_input,
    temperature=0,
    max_tokens=100,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop=["\n"]
    )

    return (((response.get('choices'))[0]).get('text'))

def query_response(user_query) :
    user_query = user_query.upper()
    query = user_query.replace('CSI', 'corporate strategy and implementation in aaruush').replace('ORM', 'Operations and Resource Management in aaruush').replace('ORM', 'Operations and Resource Management in aaruush').replace('workshops', 'Workshops in aaruush').replace('Rathinam', 'Dr. A. Rathinam').replace('RATHINAM', 'Dr. A. Rathinam , the convenor of aaruush').replace('SANJUKTA', 'Sanjukta from Team Envision and Yuddhame').replace('CREATIVES', 'creatives committee of aaruush')
    context = (get_similar_context(query)).strip()
    prompt_input_u = f'{context}'+ "This is edition i.e. A'23 is 17th edition of aaruush. act as a normal chatbot for small talks"
    query_f = ((func(prompt_input_u)).strip())
    return (get_response(((query_f + f'\n\nQ:{query}.if available give the response on the data of this edition else previous edition?\n\nA:'))))

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    user_query = data.get('user_query')
    response = query_response(user_query)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)