import pandas as pd
import json

df = pd.read_csv('data.csv')
df.head()

jsonL = []

prompt = f'''
You are to classify whether a given sentence is a pun based on the following criteria:
1. Ambiguity -  there exists a word in the sentence that has two similarly likely interpretations.
2. Distinctiveness - the two interpretations are very different from each other i.e. how distinct are the words semantically related to the two interpretations from each other. There needs to be at least one different word in the set of words supporting each interpretation.\n
For a given sentence to be a pun, it should satisfy BOTH criteria - Ambiguity and Distinctiveness.\n
The final output is either true or false where true means that the sentence is a pun.\n
It is possible that a sentence does not have a word that is ambiguous. The result in this case is false.\n
It is possible that a sentence has Ambiguity but no Distinctiveness. The result in this case is false.\n

Here are three examples of the cases you will encounter.
1. An example with both ambiguity and distinctiveness:
The magician got so mad that he pulled his hare out.
{{
  "output": true,
  "ambiguity": "The pun word 'hare' supports two plausible interpretations of 'hare' meaning a rabbit and 'hair' meaning human hair.",
  "distinctiveness": "In the given sentence, the words 'magician' relates to 'hare' while 'pulled' refers to the second interpretation of 'hair'. Both of these are distinct where one refers to a magician's animal while the other refers to an action done in anger which is pulling your hair."
}}
\n
2. An example with only ambiguity:
I went to the bank.
{{
  "output": false,
  "ambiguity": "The word 'bank' does have ambiguity here where it supports two plausible interpretations of bank as in a financial institution and bank as in the banks of a river.",
  "distinctiveness": "There are no other words in the sentence that provide distinctiveness to the two interpretations, the sentence is not a pun."
}}
\n
3. An example with neither ambiguity nor distinctiveness:
Let us go home.
{{
  "output": false,
  "ambiguity": "There is no ambiguious word in the sentence.",
  "distinctiveness": "Not Applicable"
}}
\n
Identify whether the given sentence is a pun and explain the result based on ambiguity and distinctiveness in valid JSON format.\n
'''

for _, row in df.iterrows():
  messages = []
  input_str = str(row['input'])
  output_str = str(row['output'])
  messages.append({ "role": "system", "content": prompt })
  messages.append({ "role": "user", "content": input_str })
  messages.append({ "role": "assistant", "content": output_str })
  jsonL.append({ "messages": messages })

with open("data.train.jsonl", "w") as out:
  for ddict in jsonL:
    jout = json.dumps(ddict) + "\n"
    out.write(jout)