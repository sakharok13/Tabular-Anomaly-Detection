from model import Unsuper, SimpleAutoEncoder, VAE, SemiSuper
from torch import optim
#### train for Unsuper
def train_unsuper(model, optimizer, epochs, data, batch_size, point):
  running_loss = 0 
  
  for i in range(epochs):
   for k in range(len(data) // batch_size):

    batch = data[k*batch_size:k*batch_size+batch_size]
    
    optimizer.zero_grad()
    output = model(torch.tensor(batch).float()) ##### if your data is a pandas dataframe 
    loss = criterion(output, point)

    running_loss += loss

    loss.backward()
    optimizer.step()
  print('Running loss is:', running_loss / (len(data)//batch_size) ) 
  losses.append(running_loss / (len(data)//batch_size))
  running_loss = 0




##### train for SimpleAutoEncoder
def train_ae(model, optimizer, epochs, data, batch_size):
  running_loss = 0 
  for i in range(epochs):
   for k in range(len(data) // batch_size):

    batch = data[k*batch_size:k*batch_size+batch_size]
    
    optimizer.zero_grad()
    output = model(torch.tensor(batch).float()) ##### if your data is a pandas dataframe 
    loss = criterion(output, torch.tensor(batch).float())

    running_loss += loss

    loss.backward()
    optimizer.step()
  print('Running loss is:', running_loss / (len(data)//batch_size) ) 
  losses.append(running_loss / (len(data)//batch_size))
  running_loss = 0

  
  ####### VAE TRAIN 
  
  
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
#put your dataframe instead of YOUR_DATA



from torch.functional import F
from torch.nn import MSELoss

def loss_function(recon_x, x, mu, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    mse = MSELoss()(recon_x, x)
    return mse + KLD 
  
  

def train_vae(model, optimizer, epochs, data, batch_size, device, YOUR_DATA):
  train_tensor = TensorDataset(torch.tensor(np.array(YOUR_DATA))) 
  train_loader = DataLoader(dataset = train_tensor, batch_size = 100, shuffle = True)
  for epoch in range(epochs):
    for batch_idx, data in enumerate(train_loader):
      data = data[0].to(device).float()
      recon_batch, mu, logvar = model(data)
      optimizer.zero_grad()
      loss = loss_function(recon_batch, data, mu, logvar)
      loss.backward()
      train_loss += loss.detach().cpu().numpy()
      optimizer.step()
      if batch_idx % 40 == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader),
        loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
    epoch, train_loss / len(train_loader.dataset)))      
    train_loss = 0


