import torch
from transformers import pipeline
import pandas as pd
import csv
from sklearn import metrics

df = pd.read_csv('dyn_data_test.csv', sep = '\t', doublequote=True, quotechar = "'")
def lower_str(s):
	return s.lower()

df['text'] = df.text.apply(lower_str)

comments = list(df['text'])
print('data prep done')

#byt5 (https://huggingface.co/Narrativa/byt5-base-tweet-hate-detection)
byt5 = pipeline(model = 'Narrativa/byt5-base-tweet-hate-detection')
label_list3 = byt5(comments)
labels_byt5 = [1 if i['generated_text'] == 'hate-speech' else 0 for i in label_list3]
df['byt5_pred'] = labels_byt5
print('byt5 done')

#hatexplain (https://huggingface.co/Hate-speech-CNERG/bert-base-uncased-hatexplain)
hatexplain = pipeline(model = 'Hate-speech-CNERG/bert-base-uncased-hatexplain')
hatexplain_labels = [i['label'] for i in hatexplain(comments)]
df['hatexplain_pred'] = hatexplain_labels

def binarize_1(i):
  return 1 if i == 'hate speech' else 0

df['hatexplain_pred'] = df.hatexplain_pred.apply(binarize_1)
print('hatexplain done')

#dehatebert (https://huggingface.co/Hate-speech-CNERG/dehatebert-mono-english)
dehatebert = pipeline(model = 'Hate-speech-CNERG/dehatebert-mono-english')
dehatebert_labels = [i['label'] for i in dehatebert(comments)]

df['dehatebert_pred'] = dehatebert_labels

def binarize_3(i):
  return 1 if i == 'HATE' else 0

df['dehatebert_pred'] = df.dehatebert_pred.apply(binarize_3)
print('dehatebert done')
df.to_csv('res_cross_dataset_hate.csv', sep='\t')

f1_byt5 = metrics.f1_score(df['label'], df['byt5_pred'], average = 'macro')
f1_hatexplain = metrics.f1_score(df['label'], df['hatexplain_pred'], average = 'macro')
f1_dehatebert = metrics.f1_score(df['label'], df['dehatebert_pred'], average = 'macro')
with open('f1_result.txt', 'w') as f:
    f.write('The Macro F1 scores of the models: ')
    f.write('\n')
    f.write('byt5: ' + str(f1_byt5))
    f.write('\n')
    f.write('hatexplain: ' + str(f1_hatexplain))
    f.write('\n')
    f.write('dehatebert: '+ str(f1_dehatebert))


print('all done!!')
