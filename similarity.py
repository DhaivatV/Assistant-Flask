import numpy as np
import openai
import pandas as pd
from dotenv import load_dotenv
import os
import requests

load_dotenv()

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



def search(query):

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
    
    elif similarity>0.7:
      i = data.index(similarity)
      res = fetch_row_by_question(question_list[i])
      output_string = output_string + res

    elif similarity>0.5:
      i = data.index(similarity)
      res = fetch_row_by_question(question_list[i])
      output_string = output_string + res


  return output_string



