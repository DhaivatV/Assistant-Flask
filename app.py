import os
import openai
from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from googletrans import Translator
import requests

load_dotenv()
translator = Translator()

app = Flask(__name__)
CORS(app)


openai.api_key =  os.getenv('API_KEY')

file_path = os.path.join(os.path.dirname(__file__))
final_qa_path = os.path.join(file_path, 'final_qa.csv')
df = pd.read_csv(final_qa_path)

df = pd.read_csv('final_qa.csv')

question_list = []
answer_list = []

Question_list = df['Questions'].to_list()
Answer_list = df['Answers'].to_list()


for question in Question_list:
   temp_ques = question.split(",")
   for ques in temp_ques:
      ques_temp = ques.split('\r\n')
      for final_ques in ques_temp:
        question_list.append(final_ques[2:])

def fetch_row_by_question(question):
    for index, row in df.iterrows():
        if question in row['Questions']:
            ans = (row['Answers'][2:].split('\r\n'))
            ans_str = " ".join(ans)
            return ans_str
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
              "source_sentence": query,
              "sentences": question_list
          }
      }


  )


  output_string = ""

  for similarity in data:
    if similarity>0.9:
      i = data.index(similarity)
      res = fetch_row_by_question(question_list[i])
      output_string = output_string + res

    elif similarity>=0.7:
      i = data.index(similarity)
      res = fetch_row_by_question(question_list[i])
      output_string = output_string + res
    

  if output_string != "" :
    return output_string
  
  else :
     output_string = "None"
     return output_string

     
  





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

def strip_func(value):
    return ''.join(value.splitlines())

def query_response(user_query) :
  
    context = (similar_context(user_query))
    print(context)
    if context != "None":
      final_query = (((context + f'\n\nQ:{user_query}\n\nA:')))
      res = (get_response(final_query))
      return res
    
    else:
       return("Sorry!! Please Try Again Later")


@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    user_query = data.get('user_query')
    print(user_query)
    response = query_response(user_query)
    print(response)
    return jsonify({'response': response})
    # return ("Hello")

if __name__ == '__main__':
    app.run(debug=True)