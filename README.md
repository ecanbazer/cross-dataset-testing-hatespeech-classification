## cross-dataset-testing-hatespeech-classification
Comparing Macro F1 scores of three different hate speech classification models. 

The models are tested on the test set split on Dynamically-Generated-Hate-Speech-Dataset (https://github.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset#dynamically-generated-hate-speech-dataset)

### Usage
python cross-dataset-test-hate.py -> will output the dataframe with the data, gold labels and predictions of each model as a .CSV and the F1 scores of each model as .TXT  

### The models tested

ByT5-base fine-tuned for Hate Speech Detection (on Tweets) (https://huggingface.co/Narrativa/byt5-base-tweet-hate-detection)    
BERTSequenceClassifier fine-tuned on HateXplain data (https://huggingface.co/Hate-speech-CNERG/bert-base-uncased-hatexplain)    
dehatebert (https://huggingface.co/Hate-speech-CNERG/dehatebert-mono-english)
