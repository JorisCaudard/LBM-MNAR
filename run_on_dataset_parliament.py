import torch
import json
import argparse
import numpy as np
from train_procedure import train_with_LBFGS
from lbm_nmar import LBM_NMAR
from lbfgs import FullBatchLBFGS
from utils import reparametrized_expanded_params, init_random_params

##############" LOADING Arguments" ################

parser = argparse.ArgumentParser()
parser.add_argument("--nb_row_classes", default=3)
parser.add_argument("--nb_col_classes", default=5)
parser.add_argument("--device", default='cpu')
parser.add_argument("--device2", default=None)
args = parser.parse_args()

nq = int(args.nb_row_classes)
nl = int(args.nb_col_classes)
device = torch.device('cpu') if args.device == 'cpu' else torch.device("cuda:" + str(args.device))
device2 = torch.device("cuda:" + str(args.device2)) if args.device2 else None

if not torch.cuda.is_available() and args.device != 'cpu':
    print('Cuda is not available. Algorithm will use cpu')
    device, device2 = torch.device('cpu'), None

##############" LOADING DATASET" ################

votes = np.loadtxt("data_parliament/votes.txt",delimiter=";").astype(int)
deputes = json.load(open('data_parliament/deputes.json', 'r'))
texts = json.load(open('data_parliament/texts.json', 'r'))
n1, n2 = votes.shape


##############" Init and creating model " ################
vector_of_parameters = torch.tensor(
    init_random_params(n1, n2, nq, nl), requires_grad=True, device=device
)

model = LBM_NMAR(
    vector_of_parameters,
    votes,
    (n1, n2, nq, nl),
    device=device,
    device2=device2,
)

try:
    success, loglike = train_with_LBFGS(model)
except KeyboardInterrupt:
    print("KeyboardInterrupt detected, stopping training")

# Parameters of the model
(
    nu_a,
    rho_a,
    nu_b,
    rho_b,
    nu_p,
    rho_p,
    nu_q,
    rho_q,
    tau_1,
    tau_2,
    mu_un,
    sigma_sq_a,
    sigma_sq_b,
    sigma_sq_p,
    sigma_sq_q,
    alpha_1,
    alpha_2,
    pi,
) = reparametrized_expanded_params(torch.cat((model.variationnal_params, model.model_params)), n1, n2, nq, nl, device)

# Get the row and column classes with the MAP on the varitional distributions
tau_1 = np.array(tau_1.tolist())
tau_2 = np.array(tau_2.tolist())
row_classes = tau_1.argmax(axis=1)
column_classes = tau_2.argmax(axis=1)
print("Texts classes : ", column_classes)
print("MPs classes : ", row_classes)
print([[(deputes[i]['nom'], deputes[i]['groupe']) for i, c in enumerate(row_classes) if c == nq] for nq in range(nq)])

######### Display the vote matrix, rows and columns re-ordered according to their respective classes.
import matplotlib.pyplot as plt
plt.imshow(
    votes[np.argsort(tau_1.argmax(axis=1)), :][
        :, np.argsort(tau_2.argmax(axis=1))
    ]
)
plt.show()
