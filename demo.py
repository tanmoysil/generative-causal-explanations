"""
    demo.py
    
    Reproduces Figure 3 in O'Shaughnessy et al., 'Generative causal
    explanations of black-box classifiers,' Proc. NeurIPS 2020: global
    explanation for CNN classifier trained on MNIST 3/8 digits.

    c_dim = number of features
"""

import numpy as np
import scipy.io as sio
import os
import torch
import util
import plotting
from GCE import GenerativeCausalExplainer
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from classifier import Classifier
from sklearn.preprocessing import StandardScaler
import pandas as pd

# --- parameters ---
# dataset
data_classes = [0, 1]
# classifier
#classifier_path = './pretrained_models/mnist_38_classifier'
classifier_path = "/home/visualdbs/Documents/GitHub/GCE_VAE/pretrained_models/diabetes_01_classifier"
# vae
K = 0
L = 1
vae_dim = K + L
train_steps = 1000
Nalpha = 50
Nbeta = 100
lam = 0
batch_size = 4096
lr = 5e-4
# other
randseed = 0
#gce_path = './pretrained_models/mnist_38_gce'
retrain_gce = True # train explanatory VAE from scratch
save_gce = False # save/overwrite pretrained explanatory VAE at gce_path


# --- initialize ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f" Using {device}")
# if randseed is not None:
#     np.random.seed(randseed)
#     torch.manual_seed(randseed)
# ylabels = range(0,len(data_classes))


# --- load data ---
# from load_mnist import load_mnist_classSelect
# X, Y, tridx = load_mnist_classSelect('train', data_classes, ylabels)
# vaX, vaY, vaidx = load_mnist_classSelect('val', data_classes, ylabels)
# ntrain, nrow, ncol, c_dim = X.shape
# x_dim = nrow*ncol

# iris = load_iris()
# X = iris.data[0:99]
# y = iris.target[0:99]
#
# n_features = X.shape[1]
# x_dim = X.shape[1]
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True)

dataset = 'diabetes'
class_use = np.array2string(np.array([0,1]))
save_folder_root = './pretrained_models'
save_folder = os.path.join(save_folder_root, dataset + '_' + class_use[1:(len(class_use)-1):2] + '_classifier')

df1 = pd.read_csv("/home/visualdbs/Downloads/diabetes_binary_health_indicators_BRFSS2015.csv")
df1.drop_duplicates(inplace = True)

X_pd = df1.iloc[:, 1:]
y_pd = df1.iloc[:, 0]

X = X_pd.to_numpy()
y = y_pd.to_numpy()

n_features = X.shape[1]
x_dim = X.shape[1]
n_samples = X.shape[0]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# --- load classifier ---
#from models.CNN_classifier import CNN
#classifier = CNN(len(data_classes)).to(device)
classifier =Classifier(input_dim=n_features,
                   hidden_dim1=128,
                   output_dim=1).to(device)
checkpoint = torch.load('%s/model.pt' % classifier_path, map_location=device)
classifier.load_state_dict(checkpoint['model_state_dict_classifier'])


# --- train/load GCE ---
#from models.CVAE import Decoder, Encoder
from models.VAE import Encoder, Decoder
if retrain_gce:
    # encoder = Encoder(K+L, c_dim, x_dim).to(device)
    # decoder = Decoder(K+L, c_dim, x_dim).to(device)
    encoder = Encoder(input_dim = x_dim, latent_dim = vae_dim, hidden_dim=512).to(device)
    decoder = Decoder(output_dim = x_dim, latent_dim = vae_dim, hidden_dim=512).to(device)
    #encoder.apply(util.weights_init_normal)
    #decoder.apply(util.weights_init_normal)
    gce = GenerativeCausalExplainer(classifier, decoder, encoder, device)
    traininfo = gce.train(X_train, K, L,
                          steps=train_steps,
                          Nalpha=Nalpha,
                          Nbeta=Nbeta,
                          lam=lam,
                          batch_size=batch_size,
                          lr=lr)
    if save_gce:
        if not os.path.exists(gce_path):
            os.makedirs(gce_path)
        torch.save(gce, os.path.join(gce_path,'model.pt'))
        sio.savemat(os.path.join(gce_path, 'training-info.mat'), {
            'data_classes' : data_classes, 'classifier_path' : classifier_path,
            'K' : K, 'L' : L, 'train_step' : train_steps, 'Nalpha' : Nalpha,
            'Nbeta' : Nbeta, 'lam' : lam, 'batch_size' : batch_size, 'lr' : lr,
            'randseed' : randseed, 'traininfo' : traininfo})
else: # load pretrained model
    gce = torch.load(os.path.join(gce_path,'model.pt'), map_location=device)


# --- compute final information flow ---
I = gce.informationFlow()
Is = gce.informationFlow_singledim(range(0,K+L))
print('Information flow of K=%d causal factors on classifier output:' % K)
print(Is[:K])
print('Information flow of L=%d non-causal factors on classifier output:' % L)
print(Is[K:])


# --- generate explanation and create figure ---
sample_ind = np.concatenate((np.where(y_val == 0)[0][:4],
                             np.where(y_val == 1)[0][:4],
                             np.where(y_val == 2)[0][:4]))
x = torch.from_numpy(X_val[sample_ind])
zs_sweep = [-3., -2., -1., 0., 1., 2., 3.]
Xhats, yhats = gce.explain(x, zs_sweep)
plotting.plotExplanation(1.-Xhats, yhats, save_path='figs/demo')