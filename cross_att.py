import torch
torch.manual_seed(123)
import torch.nn.functional as F

sentence = 'Life is short, eat dessert first'

dc = {s:i for i,s in enumerate(sorted(sentence.replace(',', '').split()))}
print(dc)
sentence_int = torch.tensor([dc[s] for s in sentence.replace(',','').split()])
print(sentence_int)
embed = torch.nn.Embedding(6,16)
embedded_sentence = embed(sentence_int).detach()

d = embedded_sentence.shape[1]
print(embedded_sentence.shape)

d_q, d_k, d_v = 24,24,28

W_query = torch.rand(d_q, d)
W_key = torch.rand(d_k, d)
W_value = torch.rand(d_v, d)

print(W_query.shape, W_key.shape, W_value.shape)

embedded_sentence_2 = torch.rand(8,16)

keys = W_key.matmul(embedded_sentence_2.T).T
values = W_value.matmul(embedded_sentence_2.T).T

print(keys.shape)
print(values.shape)