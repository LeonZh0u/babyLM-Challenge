import torch
from utils import *

class HMM_syllable(torch.nn.Module):
  """
  Hidden Markov Model to generate babble words
  """
  def __init__(self, M = 26, N = 6):
    super(HMM_syllable, self).__init__()
    self.M = M # number of possible observations
    self.N = N # number of states

    # A
    self.transition_model = TransitionModel(self.N)

    # b(x_t)
    self.emission_model = EmissionModel(self.N,self.M)

    # pi
    self.unnormalized_state_priors = torch.nn.Parameter(torch.randn(self.N))

    # use the GPU
    self.is_cuda = torch.cuda.is_available()
    if self.is_cuda: self.cuda()
  
  def sample(self, T=10, stop_token_index = -1):
    state_priors = torch.nn.functional.softmax(self.unnormalized_state_priors, dim=0)
    transition_matrix = torch.nn.functional.softmax(self.transition_model.unnormalized_transition_matrix, dim=0)
    emission_matrix = torch.nn.functional.softmax(self.emission_model.unnormalized_emission_matrix, dim=1)

    # sample initial state
    z_t = torch.distributions.categorical.Categorical(state_priors).sample().item()
    z = []; x = []
    z.append(z_t)
    for t in range(0,T):
      # sample emission
      x_t = torch.distributions.categorical.Categorical(emission_matrix[z_t]).sample().item()
      x.append(x_t)

      # sample transition
      z_t = torch.distributions.categorical.Categorical(transition_matrix[:,z_t]).sample().item()
      if t < T-1: z.append(z_t)

    return x, z

  def forward(self, x, T):
    """
    x : IntTensor of shape (batch size, T_max)
    T : IntTensor of shape (batch size)

    Compute log p(x) for each example in the batch.
    T = length of each example
    """
    if self.is_cuda:
      x = x.cuda()
      T = T.cuda()

    batch_size = x.shape[0]
    T_max = x.shape[1]
    log_state_priors = torch.nn.functional.log_softmax(self.unnormalized_state_priors, dim=0)
    log_alpha = torch.zeros(batch_size, T_max, self.N)
    if self.is_cuda: log_alpha = log_alpha.cuda()

    log_alpha[:, 0, :] = self.emission_model(x[:,0]) + log_state_priors
    for t in range(1, T_max):
      log_alpha[:, t, :] = self.emission_model(x[:,t]) + self.transition_model(log_alpha[:, t-1, :])

    # Select the sum for the final timestep (each x may have different length).
    log_sums = log_alpha.logsumexp(dim=2)
    log_probs = torch.gather(log_sums, 1, T.view(-1,1) - 1)
    return log_probs

  def viterbi(self, x, T):
    """
    x : IntTensor of shape (batch size, T_max)
    T : IntTensor of shape (batch size)
    Find argmax_z log p(x|z) for each (x) in the batch.
    """
    if self.is_cuda:
      x = x.cuda()
      T = T.cuda()

    batch_size = x.shape[0]
    T_max = x.shape[1]
    log_state_priors = torch.nn.functional.log_softmax(self.unnormalized_state_priors, dim=0)
    log_delta = torch.zeros(batch_size, T_max, self.N).float()
    psi = torch.zeros(batch_size, T_max, self.N).long()
    if self.is_cuda:
      log_delta = log_delta.cuda()
      psi = psi.cuda()

    log_delta[:, 0, :] = self.emission_model(x[:,0]) + log_state_priors
    for t in range(1, T_max):
      max_val, argmax_val = self.transition_model.maxmul(log_delta[:, t-1, :])
      log_delta[:, t, :] = self.emission_model(x[:,t]) + max_val
      psi[:, t, :] = argmax_val

    # Get the log probability of the best path
    log_max = log_delta.max(dim=2)[0]
    best_path_scores = torch.gather(log_max, 1, T.view(-1,1) - 1)

    # This next part is a bit tricky to parallelize across the batch,
    # so we will do it separately for each example.
    z_star = []
    for i in range(0, batch_size):
      z_star_i = [ log_delta[i, T[i] - 1, :].max(dim=0)[1].item() ]
      for t in range(T[i] - 1, 0, -1):
        z_t = psi[i, t, z_star_i[0]].item()
        z_star_i.insert(0, z_t)

      z_star.append(z_star_i)

    return z_star, best_path_scores # return both the best path and its log probability

class TransitionModel(torch.nn.Module):
  def __init__(self, N):
    super(TransitionModel, self).__init__()
    self.N = N
    self.unnormalized_transition_matrix = torch.nn.Parameter(torch.randn(N,N))

  def forward(self, log_alpha):
    """
    log_alpha : Tensor of shape (batch size, N)
    Multiply previous timestep's alphas by transition matrix (in log domain)
    """
    log_transition_matrix = torch.nn.functional.log_softmax(self.unnormalized_transition_matrix, dim=0)

    # Matrix multiplication in the log domain
    out = log_domain_matmul(log_transition_matrix, log_alpha.transpose(0,1)).transpose(0,1)
    return out

class EmissionModel(torch.nn.Module):
  def __init__(self, N, M):
    super(EmissionModel, self).__init__()
    self.N = N
    self.M = M
    self.unnormalized_emission_matrix = torch.nn.Parameter(torch.randn(N,M))

  def forward(self, x_t):
    log_emission_matrix = torch.nn.functional.log_softmax(self.unnormalized_emission_matrix, dim=1)
    out = log_emission_matrix[:, x_t].transpose(0,1)
    return out
