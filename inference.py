from trainer import *
from hmm_syllables import *
from utils import *
import numpy as np
import pandas as pd
import seaborn as sns
from fitter import Fitter, get_common_distributions, get_distributions
import pickle
from scipy.stats import lognorm


"""
with open("sentences", "rb") as fp:   # Unpickling
      sentences = pickle.load(fp)
lenDist = list(map(lambda x: len(word_tokenize(x)), sentences))
sns.set_style('white')
sns.set_context("paper", font_scale = 2)
sns.displot(data=lenDist, kind="hist", bins = 100, aspect = 1.5)
f = Fitter(lenDist,
           distributions=
['alpha', 'anglit', 'arcsine', 'argus', 'beta', 'betaprime', 'bradford', 'burr', 'burr12', 'cauchy', 'chi', 'chi2', 'cosine', 'crystalball', 'dgamma', 'dweibull', 'erlang', 'expon', 'exponnorm', 'exponpow', 'exponweib', 'f', 'fatiguelife', 'fisk', 'foldcauchy', 'foldnorm', 'frechet_l', 'frechet_r', 'gamma', 'gausshyper', 'genexpon', 'genextreme', 'gengamma', 'genhalflogistic', 'geninvgauss', 'genlogistic', 'gennorm', 'genpareto', 'gilbrat', 'gompertz', 'gumbel_l', 'gumbel_r', 'halfcauchy', 'halfgennorm', 'halflogistic', 'halfnorm',
'hypsecant', 'invgamma', 'invgauss', 'invweibull', 'johnsonsb', 'johnsonsu', 'kappa3', 'kappa4', 'ksone', 'kstwo', 'kstwobign', 'laplace', 'levy', 'levy_l', 'levy_stable', 'loggamma', 'logistic', 'loglaplace', 'lognorm', 'loguniform', 'lomax', 'maxwell', 'mielke', 'moyal', 'nakagami', 'ncf', 'nct', 'ncx2',
'norm', 'norminvgauss', 'pareto', 'pearson3', 'powerlaw', 'powerlognorm', 'powernorm', 'rayleigh', 'rdist', 'recipinvgauss', 'reciprocal', 'rice', 'rv_continuous', 'rv_histogram', 'semicircular', 'skewnorm', 't', 'trapz', 'triang', 'truncexpon', 'truncnorm', 'tukeylambda', 'uniform', 'vonmises', 'vonmises_line', 'wald', 'weibull_max', 'weibull_min', 'wrapcauchy']
)
f.fit()
print(f.summary())
print(f.get_best(method = 'sumsquare_error'))
"""
sample_size = 10
sampled_lens = list(map(int, lognorm.rvs(0.5491595459713867, loc = -0.4303603242253272, scale = 4.628530660114688, size = sample_size)))

v2 = torch.load('vocab_obj.pth')
v2.insert_token(" ", len(v2))
v2.set_default_index(v2['<unk>'])

model = HMM_syllable(M=len(v2), N=6) 
model.load_state_dict(torch.load("model_tokens.pt"))
model.eval()
for i in range(sample_size):
      sampled_x, sampled_z = model.sample(T = sampled_lens[i])
      print(sampled_z, decode(sampled_x, v2))
