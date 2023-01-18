import torch
from tqdm import tqdm 
    
class Trainer:
  def __init__(self, model, lr, encoder = None, decoder = None):
    self.model = model
    self.lr = lr
    self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=0.00001)
    self.encode = encoder
    self.decode = decoder

  def train(self, dataset):
    train_loss = 0
    num_samples = 0
    self.model.train()
    print_interval = 50
    for idx, batch in enumerate(tqdm(dataset.loader)):
      x,T = batch
      batch_size = len(x)
      num_samples += batch_size
      log_probs = self.model(x,T)
      loss = -log_probs.mean()
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
      train_loss += loss.cpu().data.numpy().item() * batch_size
      if idx % print_interval == 0:
        print("loss:", loss.item())
        for _ in range(5):
          sampled_x, sampled_z = self.model.sample()
          print(self.decode(sampled_x))
          print(sampled_z)
    train_loss /= num_samples
    return train_loss

  def test(self, dataset):
    test_loss = 0
    num_samples = 0
    self.model.eval()
    print_interval = 50
    for idx, batch in enumerate(dataset.loader):
      x,T = batch
      batch_size = len(x)
      num_samples += batch_size
      log_probs = self.model(x,T)
      loss = -log_probs.mean()
      test_loss += loss.cpu().data.numpy().item() * batch_size
      if idx % print_interval == 0:
        print("loss:", loss.item())
        sampled_x, sampled_z = self.model.sample()
        print(self.decode(sampled_x))
        print(sampled_z)
    test_loss /= num_samples
    return test_loss
