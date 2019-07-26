#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from pathlib import Path
import seaborn as sns
from fastai.text import *
from fastai import __version__


# In[2]:


print("Python version: {}". format(sys.version))
print("seaborn version: {}". format(sns.__version__))
print("fastai version: {}". format(__version__))


# # Preparando os dados

# In[3]:


# setando o path para os dados
path = Path('./data')


# In[ ]:


df = pd.read_csv(path/'treinamento.csv', encoding = 'ISO-8859-1')


# In[ ]:


df.shape


# In[ ]:


df.isnull().sum()


# In[ ]:


df.head()


# In[ ]:


df['categoria'].value_counts()


# In[ ]:


sns.set(style="darkgrid")
ax = sns.countplot(x="categoria", data=df)
plt.show()


# In[ ]:


df.to_csv(path/'treinamento_utf-8.csv', encoding = 'utf-8')


# ### Criação de um objeto "Databunch" para realizar "Tokenization" e "Numericalization" dos dados.

# In[ ]:


data = TextDataBunch.from_csv(path, 'treinamento_utf-8.csv', text_cols='mensagem', label_cols='categoria')


# ### Tokenization

# In[ ]:


data.show_batch()


# In[ ]:


data.train_ds[0][0]


# ### Numericalization

# In[ ]:


data.vocab.itos[:18]


# In[ ]:


data.train_ds[0][0].data[:10]


# # Modelo de Linguagem

# In[5]:


bs=48


# In[ ]:


tokenizer = Tokenizer(SpacyTokenizer, 'pt')
processor = [TokenizeProcessor(tokenizer=tokenizer), NumericalizeProcessor(max_vocab=30000)]


# ### Criando o DataBunch 

# In[ ]:


data_lm = (TextList.from_csv(path, 'treinamento_utf-8.csv', cols='mensagem', processor=processor)           
            .split_by_rand_pct(0.1, seed=50) #Dividindo randomicamente o dataset em 10% para validação
            .label_for_lm()                  ##Setando a coluna de label
            .databunch(bs=bs))
data_lm.save('data_lm.pkl')


# In[6]:


data_lm = load_data(path, 'data_lm.pkl', bs=bs)


# In[10]:


data_lm.show_batch()


# ### Criando o modelo de linguagem
# 
# A partir de um modelo de linguagem "pré-treinado", será criado um modelo customizado para as mensagens.

# In[ ]:


# Nomes dos arquivos de modelo e vocabulário pré-treinado
pretrained_fnames = ('lm_Pt_Br_30kt_ft', 'itos')
learn = language_model_learner(data_lm, AWD_LSTM, pretrained_fnames=pretrained_fnames, drop_mult=0.3)


# In[12]:


learn.lr_find()


# In[13]:


learn.recorder.plot(skip_end=3)


# ### Treinando o modelo 

# In[14]:


learn.fit_one_cycle(1, 1e-1, moms=(0.8,0.7))


# In[ ]:


learn.unfreeze()


# In[16]:


learn.fit_one_cycle(10, 1e-2, moms=(0.8,0.7))


# In[ ]:


learn.save('lm_reviews_ft')


# In[ ]:


learn.load('lm_reviews_ft');


# ### Testando o modelo
# 
# Testando as previsões realizadas pelo modelo de próximas palavras de um texto inserido.

# In[ ]:


TEXT = "Depois de realizar"
N_WORDS = 18


# In[22]:


learn.predict(TEXT, N_WORDS, temperature=0.75)


# ### Salvando o encoder para utilização no classificador

# In[ ]:


learn.save_encoder('lm_reviews_ft_enc')


# # Classificador

# In[ ]:


bs=48


# In[ ]:


tokenizer = Tokenizer(SpacyTokenizer, 'pt')
processor = [TokenizeProcessor(tokenizer=tokenizer), NumericalizeProcessor(max_vocab=30000)]


# In[ ]:


data_clas = (TextList.from_csv(path, 'treinamento_utf-8.csv', cols='mensagem', vocab=data_lm.vocab, processor=processor)           
            .split_by_rand_pct(0.1, seed=50)       #Dividindo randomicamente o dataset em 10% para validação
            .label_from_df(cols='categoria')       #Setando a coluna de label 
            .databunch(bs=bs))
data_clas.save('data_clas.pkl')


# In[7]:


data_clas = load_data(path, 'data_clas.pkl', bs=bs)


# In[79]:


data_clas.show_batch()


# ### Criando o modelo
# 
# Criando o modelo para classificação das mensagens e carregando o encoder salvo anteriormente.

# In[39]:


wgts_fname = path/'models'/'lm_Pt_Br_30kt_ft.pth'
itos_fname = path/'models'/'itos.pkl'

model = get_text_classifier(AWD_LSTM, len(data_clas.vocab.itos), data_clas.c, drop_mult=0.5)
learn = RNNLearner(data_clas, model, split_func=awd_lstm_clas_split) 
learn.load_pretrained(wgts_fname, itos_fname, strict=False)
learn.freeze()


# In[40]:


learn.load_encoder('lm_reviews_ft_enc')


# In[20]:


learn.lr_find()


# In[21]:


learn.recorder.plot()


# In[22]:


lr=2e-2
lr *= bs/48


# ### Treinando o modelo

# In[23]:


learn.fit_one_cycle(2, lr, moms=(0.8,0.7))


# In[24]:


learn.freeze_to(-2)
learn.fit_one_cycle(2, slice(lr/(2.6**4),lr), moms=(0.8,0.7))


# In[27]:


learn.freeze_to(-3)
learn.fit_one_cycle(2, slice(lr/2/(2.6**4),lr/2), moms=(0.8,0.7))


# In[42]:


learn.unfreeze()
learn.fit_one_cycle(1, slice(lr/10/(2.6**4),lr/10), moms=(0.8,0.7))


# #### Salvando o modelo com acurácia de 91%.

# In[43]:


learn.save('clas_reviews_ft')


# ### Testando o modelo 
# 
# Testando a classificação realizada pelo modelo de uma mensagem inserida. 

# In[48]:


learn.predict("O atendimento foi muito bom")

