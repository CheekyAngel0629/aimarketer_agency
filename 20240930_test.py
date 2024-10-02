import openai

model_list = sorted([m['id'] for m in openai.Model.list()['data']])
for m in model_list:
    print(m)