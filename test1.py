import requests
import json
import pandas as pd
import os
import openai
import requests
from dotenv import load_dotenv

load_dotenv()

openai.api_key =  os.getenv('API_KEY')

with open('data.json', 'r') as file:
    data = json.load(file)


df = pd.DataFrame(data['details']) 
d_c_list = df['name'].to_list()
domain_list = d_c_list[:10]
committee_list = d_c_list[10:23]
for i in range(len(df)):
    name = df.loc[i, 'name']
    if name in domain_list:
        df.loc[i, 'domain/committee'] = "domain"
    elif name in committee_list :
        df.loc[i, 'domain/committee'] = "committee"
    else:
        df.loc[i, 'domain/committee'] = "team"

df['combined_info'] = df['name'] + " " + df['domain/committee'] + " " + df['organizers'] + " " + df['description'] 
list_info = df['combined_info'].to_list()



def fetch_row_by_info(info):
    for index, row in df.iterrows():
        if info in row['combined_info']:
            res = row.to_dict()
            return res
    return None

def similar_context(query):

  API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
  api_token = "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"
  headers = {"Authorization": f"Bearer {api_token}"}

  def req(payload):
      response = requests.post(API_URL, headers=headers, json=payload)
      return response.json()

  data = req(
      {
          "inputs": {
              "source_sentence": query.lower(),
              "sentences": list_info
          }
      }


  )
              
  smi_index = []

  for similarity in data:
    if similarity>0.9:
       smi_index.append(similarity)
    
    
    elif similarity>0.7:
       smi_index.append(similarity)

    elif similarity>0.5:
       smi_index.append(similarity)

    elif similarity>0.4:
       smi_index.append(similarity)


  if len(smi_index) ==1:
     exact_info_index = data.index(smi_index[0])

  elif len(smi_index) >1:
     smi_index.sort() 
     exact_info_index = data.index(smi_index[-1])

  else:
      return("Sorry!! Not enough info provided in question")

 
      
  return json.dumps(fetch_row_by_info(list_info[exact_info_index] ))

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
  
    context = (similar_context(user_query))
    if context == "Sorry!! Not enough info provided in question":
        return (context)
    else:
        final_query = (((context + f'\n\nQ:{user_query}\n\nA:')))
        res = (get_response(final_query))
        return res
    
          

def query(user_query):
    response = query_response(user_query)
    print(response)


query("what is konstruction and canoe challenge ?")