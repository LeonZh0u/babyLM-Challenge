from trainer import *
from hmm import *
from babylm import *

model = HMM_syllable(M=2743, N=6) 
model.load_state_dict(torch.load("model.pt"))
model.eval()
print("!!!")
for _ in range(15):
      sampled_x, sampled_z = model.sample(stop_token_index = len(v2)-1)
      print(sampled_z, decode(sampled_x))
