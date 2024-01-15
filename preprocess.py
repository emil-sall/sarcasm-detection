import pandas as pd
import numpy as np
from spacy.language import Language
from spacy import load
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Implement the tensor2attr component
# Source: https://applied-language-technology.mooc.fi/html/notebooks/part_iii/05_embeddings_continued.html
@Language.factory('tensor2attr')

class Tensor2Attr:
    def __init__(self, name, nlp):
        pass
    
    def __call__(self, doc):
        self.add_attributes(doc)
        return doc
    
    def add_attributes(self, doc):
        doc.user_hooks['vector'] = self.doc_tensor
        doc.user_span_hooks['vector'] = self.span_tensor
        doc.user_token_hooks['vector'] = self.token_tensor
        doc.user_hooks['similarity'] = self.get_similarity
        doc.user_span_hooks['similarity'] = self.get_similarity
        doc.user_token_hooks['similarity'] = self.get_similarity
    
    def doc_tensor(self, doc):
        return doc._.trf_data.tensors[-1].mean(axis=0)
    
    def span_tensor(self, span):
        tensor_ix = span.doc._.trf_data.align[span.start: span.end].data.flatten()
        out_dim = span.doc._.trf_data.tensors[0].shape[-1]
        tensor = span.doc._.trf_data.tensors[0].reshape(-1, out_dim)[tensor_ix]
        return tensor.mean(axis=0)
    
    def token_tensor(self, token):
        tensor_ix = token.doc._.trf_data.align[token.i].data.flatten()
        out_dim = token.doc._.trf_data.tensors[0].shape[-1]
        tensor = token.doc._.trf_data.tensors[0].reshape(-1, out_dim)[tensor_ix]
        return tensor.mean(axis=0)

    def get_similarity(self, doc1, doc2):
        return np.dot(doc1.vector, doc2.vector) / (doc1.vector_norm * doc2.vector_norm)
    
# Preprocess the data 
def preprocess(text):
    if isinstance(text, str):
        return nlp(text)
    else:
        return nlp("")

# Extract the contextualized embeddings
def contextualize(df_row):
    embeddings_text = df_row['text_doc']._.trf_data.tensors[-1]
    embeddings_text = torch.from_numpy(embeddings_text).squeeze(0)

    if df_row['context_doc']:
        embeddings_context = df_row['context_doc']._.trf_data.tensors[-1]
        embeddings_context = torch.from_numpy(embeddings_context).squeeze(0)
    else:
        embeddings_context = torch.zeros_like(embeddings_text).squeeze(0)
    
    if embeddings_text.size() != torch.Size([768]):
        embeddings_text = embeddings_text.mean(dim=0, keepdim=True).squeeze(0)

    if embeddings_context.size() != torch.Size([768]):
        embeddings_context = embeddings_context.mean(dim=0, keepdim=True).squeeze(0)

    return embeddings_text, embeddings_context

# Load the data and remove rows with missing 'text'
data = pd.read_csv("complete_data.csv")
data = data.dropna(subset=['text'])

# Get the distribution of sarcastic labels
plt.figure()
sns.countplot(x='label', data=data).set(title='Sarcastic labels')
plt.savefig('sarcastic_labels_plot.png')

# Load the language pipeline and add the tensor2attr component
nlp = load('en_core_web_trf')
nlp.add_pipe('tensor2attr')

# Save the data in batches due to memory constraints
batch_size = 10000
nr_batches = int(np.ceil(len(data) / batch_size))

for i in range(nr_batches):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, len(data))

    batch_data = data.iloc[start_idx:end_idx]

    batch_data['text_doc'] = batch_data['text'].apply(preprocess)
    batch_data['context_doc'] = batch_data['context'].apply(preprocess)
    batch_data[['text_embeddings', 'context_embeddings']] = batch_data.apply(contextualize, axis=1).apply(pd.Series)
    batch_data['text_embeddings'] = batch_data['text_embeddings'].apply(lambda x: x.tolist())
    batch_data['context_embeddings'] = batch_data['context_embeddings'].apply(lambda x: x.tolist())

    batch_filename = f'processed_data{i}.csv'
    batch_data[['text_embeddings', 'context_embeddings', 'label']].to_csv(batch_filename, encoding='utf-8', index=False)

    print(f'Batch {i} saved successfully.')
