from nltk.tokenize import SyllableTokenizer
from nltk import word_tokenize
import nltk
import os
nltk.download('punkt')
import pandas as pd
from fitter import Fitter, get_common_distributions, get_distributions
import pickle
from tqdm import tqdm
import multiprocessing
import seaborn as sns

if __name__ == '__main__':  
    with open("data/babylm_data/babylm_10M/sentences", "rb") as fp:   # Unpickling
        sentences = pickle.load(fp)
    print("loaded")



    SSP = SyllableTokenizer()
    lenDist = []
    all_sents = []
    pool = multiprocessing.Pool(None)

    if os.path.exists('all_sents'):
        with open("all_sents", "rb") as fp:   # Unpickling
            all_sents = pickle.load(fp)
    else:
        for out in tqdm(pool.map(word_tokenize, sentences), total = len(sentences)):
            all_sents.append(out)
        with open("all_sents", "wb") as fp:  # Pickling
                pickle.dump(all_sents, fp)

    print(all_sents[:10])
    for sent in tqdm(all_sents , total = len(all_sents)):
        lens = 0
        for word in sent:
            lens+=len(SSP.tokenize(word))
        lenDist.append(lens)

    sns.set_style('white')
    sns.set_context("paper", font_scale = 2)
    plot = sns.displot(data=lenDist, kind="hist", bins = 100, aspect = 1.5)
    plot.savefig('sent_dist.png')

    f = Fitter(lenDist,
            distributions=['alpha', 'anglit', 'arcsine', 'argus', 'beta', 'betaprime', 'bradford', 'burr', 'burr12', 'cauchy', 'chi', 'chi2', 'cosine', 'crystalball', 'dgamma', 'dweibull', 'erlang', 'expon', 'exponnorm', 'exponpow', 'exponweib', 'f', 'fatiguelife', 'fisk', 'foldcauchy', 'foldnorm', 'frechet_l', 'frechet_r', 'gamma', 'gausshyper', 'genexpon', 'genextreme', 'gengamma', 'genhalflogistic', 'geninvgauss', 'genlogistic', 'gennorm', 'genpareto', 'gilbrat', 'gompertz', 'gumbel_l', 'gumbel_r', 'halfcauchy', 'halfgennorm', 'halflogistic', 'halfnorm',
    'hypsecant', 'invgamma', 'invgauss', 'invweibull', 'johnsonsb', 'johnsonsu', 'kappa3', 'kappa4', 'ksone', 'kstwo', 'kstwobign', 'laplace', 'levy', 'levy_l', 'levy_stable', 'loggamma', 'logistic', 'loglaplace', 'lognorm', 'loguniform', 'lomax', 'maxwell', 'mielke', 'moyal', 'nakagami', 'ncf', 'nct', 'ncx2',
    'norm', 'norminvgauss', 'pareto', 'pearson3', 'powerlaw', 'powerlognorm', 'powernorm', 'rayleigh', 'rdist', 'recipinvgauss', 'reciprocal', 'rice', 'rv_continuous', 'rv_histogram', 'semicircular', 'skewnorm', 't', 'trapz', 'triang', 'truncexpon', 'truncnorm', 'tukeylambda', 'uniform', 'vonmises', 'vonmises_line', 'wald', 'weibull_max', 'weibull_min', 'wrapcauchy']
    )
    f.fit()
    print(f.summary())
    print(f.get_best(method = 'sumsquare_error'))

"""
              sumsquare_error           aic           bic  kl_div  ks_statistic  ks_pvalue
gumbel_r         5.097331e-07  32940.534681 -3.002792e+07     inf      0.150661        0.0
moyal            1.072636e-06  28725.655520 -2.924023e+07     inf      0.135545        0.0
halflogistic     2.134080e-06  28682.128291 -2.851191e+07     inf      0.178010        0.0
hypsecant        2.223748e-06  33527.837830 -2.846833e+07     inf      0.218413        0.0
logistic         3.604312e-06  38098.239069 -2.795703e+07     inf      0.217713        0.0
{'gumbel_r': {'loc': 7.33330589149, 'scale': 9.011387779280227}}
"""