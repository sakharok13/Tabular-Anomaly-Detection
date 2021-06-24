from tqdm import tqdm
import matplotlib.pyplot as plt

# methods should be named as SemiSuper, VAE, Unsuper, AE for SemiSupervised, Variational AutoEncoder, SimpleUnsupervised, AutoEncoder respectively 

def plot_roc_auc_improvement(model, train, test, method = 'AE', point = None) 
  from sklearn.metrics import roc_auc_score

  if method == 'Unsuper' or method == 'SemiSuper':
    anom = torch.sum((model(torch.tensor(np.array(train)).float()) - point, axis=1)
  elif method == 'AE' or method == 'VAE':
    anom = torch.sum((model(torch.tensor(np.array(train)).float()) - torch.tensor(np.array(train)).float())**2, axis=1)

  train['anom'] = anom
  train = train.sort_values('anom')
  del train['anom']
  y_train = train['ans']
  del train['ans']
  X_train = train
  y_test = test['ans']
  del test['ans']
  X_test = test
  from sklearn.linear_model import LogisticRegression
  scores = []

  for i in tqdm(range(250)):
    lr = LogisticRegression()
    lr.fit(X_train.iloc[i:], y_train.iloc[i:])
    scores.append(roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1]))

  plt.figure(figsize = (8, 6))
  plt.title('ROC-AUC после удаления аномалий на сырых данных')
  plt.xlabel('Количество удаленных аномалий')
  plt.ylabel('ROC-AUC')
  plt.plot(scores)
  plt.show()
