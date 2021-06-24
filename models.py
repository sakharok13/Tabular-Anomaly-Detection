from torch import nn 
import torch 
#SIMPLE UNSUPERVISED METHOD THAT MAPS EVERYTHING TO A POINT 

class Unsuper(nn.Module):
  def __init__(self, n_features, point_size):
    super(Unsuper, self).__init__()
    self.map = nn.Sequential(
                              nn.Linear(n_features, n_features // 2),
                              nn.ReLU(),
      
                              nn.Linear(n_features // 2, n_features // 4),
                              nn.ReLU(),
      
                              nn.Linear(n_features // 4, point_size)
    )
    
  def forward(self, x):
    return(self.map(x))

### Simple autoencoder 

class SimpleAutoEncoder(nn.Module):
  def __init__(self, n_features):
    super(SimpleAutoEncoder, self).__init__()
    self.linear1 = nn.Sequential(
        nn.Linear(n_features, n_features // 2),
        nn.LeakyReLU(),

        nn.Linear(n_features // 2, n_features // 8),
        nn.LeakyReLU()

    )
    self.linear2 = nn.Sequential(
        nn.Linear(n_features // 8, n_features // 2),

        nn.LeakyReLU(),
        nn.Linear(n_features // 2, n_features),


    )

  def forward(self, x):
    x = self.linear1(x)
    x = self.linear2(x)
    return(x)

  
######### VARIATIONAL AUTOENCODER
class VAE(nn.Module):
    def __init__(self, feature_size, latent_size):
        super(self, feature_size).__init__()
        self.feature_size = feature_size
        #self.class_size = class_size

        self.fc1  = nn.Sequential(nn.Linear(feature_size, feature_size * 4),
                                  nn.ELU(),

                                  nn.Linear(feature_size * 4, feature_size),
                                  nn.ELU(),

                                  nn.Linear(feature_size, feature_size // 8)
        )
        self.fc21 = nn.Linear(feature_size // 8, 2)   #### change 2 to whatever you want (2 is used to visualize hidden representation)
        self.fc22 = nn.Linear(feature_size // 8, 2)

   
        self.fc3 = nn.Linear(2, feature_size // 8)
        self.fc4 = nn.Sequential(nn.Linear(feature_size // 8, feature_size),
                                 nn.ELU(),

                                 nn.Linear(feature_size, feature_size * 4),
                                 nn.ELU(),

                                 nn.Linear(feature_size * 4, feature_size)
        )

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x): 

        inputs = x
        h1 = self.elu(self.fc1(inputs))
        z_mu = self.fc21(h1)
        z_var = self.fc22(h1)
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z): 
        inputs = z
        h3 = self.elu(self.fc3(inputs))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.feature_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

     
#####SEMI SUPERVISED MODEL IS BASICALLY JUST A SIMPLE UNSUPERVISED MODEL WITH UPGRADED LOSS FUNCTION 


class SemiSuper(nn.Module):
  def __init__(self, n_features, point_size):
    super(SemiSuper, self).__init__()
    self.map = nn.Sequential(
                              nn.Linear(n_features, n_features // 2),
                              nn.ReLU(),

                              nn.Linear(n_features // 2, n_features // 4),
                              nn.ReLU(),

                              nn.Linear(n_features // 4, point_size)
    )

  def forward(self, x):
    return(self.map(x))

 
  
