import os
import streamlit as st
from llama_cpp import Llama
from openai import OpenAI
import json
from dotenv import load_dotenv
import random

load_dotenv()

MODEL = 'models/mistral-7b-openorca.Q6_K.gguf'
# API_MODEL = 'gpt-4-1106-preview'
API_MODEL = 'gpt-3.5-turbo-1106'

prompt_openocra = f''' ### INSTRUCTION:
You are to classify whether a given sentence is a pun based on the following criteria:
1. Ambiguity -  there exists a word in the sentence that has two similarly likely interpretations.
2. Distinctiveness - the two interpretations are very different from each other i.e. how distinct are the words semantically related to the two interpretations from each other. There needs to be at least one different word in the set of words supporting each interpretation.\n
For a given sentence to be a pun, it should satisfy BOTH criteria - Ambiguity and Distinctiveness.\n
The final output is either true or false where true means that the sentence is a pun.\n
It is possible that a sentence does not have a word that is ambiguous. The result in this case is false.\n
It is possible that a sentence has Ambiguity but no Distinctiveness. The result in this case is false.\n
Here are three examples of the cases you will encounter:
1. An example with both ambiguity and distinctiveness - 
The magician got so mad that he pulled his hare out.
{{
  "output": true,
  "ambiguity": "The pun word 'hare' supports two plausible interpretations of 'hare' meaning a rabbit and 'hair' meaning human hair.",
  "distinctiveness": "In the given sentence, the words 'magician' relates to 'hare' while 'pulled' refers to the second interpretation of 'hair'. Both of these are distinct where one refers to a magician's animal while the other refers to an action done in anger which is pulling your hair."
}}
2. An example with only ambiguity -
I went to the bank.
{{
  "output": false,
  "ambiguity": "The word 'bank' does have ambiguity here where it supports two plausible interpretations of bank as in a financial institution and bank as in the banks of a river.",
  "distinctiveness": "There are no other words in the sentence that provide distinctiveness to the two interpretations, the sentence is not a pun."
}}
3. An example with neither ambiguity nor distinctiveness -
Let us go home.
{{
  "output": false,
  "ambiguity": "There is no ambiguous word in the sentence.",
  "distinctiveness": "Not applicable"
}}
Identify whether the input sentence is a pun and explain the result based on ambiguity and distinctiveness in valid JSON format. Generate a response in the form of a valid JSON object with three keys: output, ambiguity and distinctiveness.\n
### INPUT:\n
'''

def prepare_prompt_gpt_humor(sentence):
  messages=[
  { "role": "system",
    "content": f'''
You are to classify whether a given sentence is a pun based on the following criteria:
1. Ambiguity -  there exists a word in the sentence that has two similarly likely interpretations.
2. Distinctiveness - the two interpretations are very different from each other i.e. how distinct are the words semantically related to the two interpretations from each other. There needs to be at least one different word in the set of words supporting each interpretation.\n
For a given sentence to be a pun, it should satisfy BOTH criteria - Ambiguity and Distinctiveness.\n
The final output is either true or false where true means that the sentence is a pun.\n
It is possible that a sentence does not have a word that is ambiguous. The result in this case is false.\n
It is possible that a sentence has Ambiguity but no Distinctiveness. The result in this case is false.\n
Here are three examples of the cases you will encounter:
1. An example with both ambiguity and distinctiveness - 
The magician got so mad that he pulled his hare out.
{{
  "output": true,
  "ambiguity": "The pun word 'hare' supports two plausible interpretations of 'hare' meaning a rabbit and 'hair' meaning human hair.",
  "distinctiveness": "In the given sentence, the words 'magician' relates to 'hare' while 'pulled' refers to the second interpretation of 'hair'. Both of these are distinct where one refers to a magician's animal while the other refers to an action done in anger which is pulling your hair."
}}
2. An example with only ambiguity -
I went to the bank.
{{
  "output": false,
  "ambiguity": "The word 'bank' does have ambiguity here where it supports two plausible interpretations of bank as in a financial institution and bank as in the banks of a river.",
  "distinctiveness": "There are no other words in the sentence that provide distinctiveness to the two interpretations, the sentence is not a pun."
}}
3. An example with neither ambiguity nor distinctiveness -
Let us go home.
{{
  "output": false,
  "ambiguity": "There is no ambiguous word in the sentence.",
  "distinctiveness": "Not applicable"
}}
Identify whether the input sentence is a pun and explain the result based on ambiguity and distinctiveness in valid JSON format. Generate a response in the form of a valid JSON object with three keys: output, ambiguity and distinctiveness.\n
'''
  },
  {
    "role": "user", "content": f'''{sentence}'''
  }
]
  return messages

@st.cache_resource
def create_model(model):
    llm = None
    if model == 'openocra-7b':
      llm = Llama(MODEL, n_gpu_layers=75, n_ctx=2048)
    else:
      llm = OpenAI(api_key=os.getenv("OPEN_API_KEY"))
    return llm

def perform_gpt(llm, sentence):
  prompt = prepare_prompt_gpt_humor(sentence)
  response = llm.chat.completions.create(
  model=API_MODEL,
  response_format={ "type": "json_object" },
  temperature=0.2,
  messages=prompt,
  timeout=5,
  )
  parsedJson = json.loads(response.choices[0].message.content)
  return parsedJson

def generate(llm, prompt, attempt):
  if attempt == 1:
    temp = 0.2
  else:
    temp = round(random.uniform(0.3, 0.6), 1)
  output = llm(prompt, max_tokens=-1, temperature=temp)
  return output['choices'][0]['text']

def is_json(strJson):
  try:
    json.loads(strJson)
  except ValueError as e:
    return False
  return True

def parseResult(jsonStr):
  if not is_json(jsonStr.strip()):
    return None
  else:
    parsedJson = json.loads(jsonStr.strip())
    return parsedJson

def perform_openocra(llm, prompt, sentence):
  prepared_prompt = prompt + sentence + "\n### OUTPUT:\n"
  json_out = ''
  attempt = 0
  while True:
    attempt += 1
    json_out = generate(llm, prepared_prompt, attempt)
    parsedJson = parseResult(json_out)
    return parsedJson

st.set_page_config(page_title="PUNderstand")
st.header("Using Large Language Models to Understand Puns")

option = st.selectbox('Which model should be used?', ('gpt4', 'openocra-7b'))
llm = create_model(option)
sentence = st.text_input('Please enter input sentence', '')
if sentence:
  result = None
  if option == 'openocra-7b':
    result = perform_openocra(llm, prompt_openocra, sentence)
  else:
    print('hit')
    result = perform_gpt(llm, sentence)
  if result:
    st.write(result)
