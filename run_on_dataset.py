import torch
import json
import argparse
import numpy as np
from LBM_MNAR_origin.train_procedure import train_with_LBFGS
from LBM_MNAR_origin.lbm_nmar import LBM_NMAR
from LBM_MNAR_origin.utils import reparametrized_expanded_params, init_random_params
from generate_dataset import plot_data_LBM, calculate_missing_proportions

##############" LOADING Arguments" ################

parser = argparse.ArgumentParser()
parser.add_argument("--nb_row_classes", default=3)
parser.add_argument("--nb_col_classes", default=3)
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

X = np.loadtxt("synth_dataset/X_mnar.csv",delimiter=";").astype(int)
n1, n2 = X.shape


##############" Init and creating model " ################
vector_of_parameters = torch.tensor(
    init_random_params(n1, n2, nq, nl), requires_grad=True, device=device
)

model = LBM_NMAR(
    vector_of_parameters,
    X,
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
est_row_classes = tau_1.argmax(axis=1)
est_column_classes = tau_2.argmax(axis=1)
print("Estimated column classes : ", est_column_classes)
print("Estimated row classes : ", est_row_classes)

######### Display the vote matrix, rows and columns re-ordered according to their respective classes.
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)
plot_data_LBM(X, est_row_classes, est_column_classes, ax, f"Original Data, Missing proportion = {calculate_missing_proportions(X):.2%}")
plt.show()

print("alpha = ", alpha_1.detach().cpu().numpy().reshape(-1))
print("beta = ", alpha_2.detach().cpu().numpy())
print("pi = ", pi.detach().cpu().numpy())
print("mu = ", mu_un.detach().cpu().numpy())
print("sigma_A = ", sigma_sq_a.detach().cpu().numpy())
print("sigma_B = ", sigma_sq_b.detach().cpu().numpy())
print("sigma_C = ", sigma_sq_p.detach().cpu().numpy())
print("sigma_D = ", sigma_sq_q.detach().cpu().numpy())

print("ICL = ", -loglike - ((nl-1)/2)*np.log(n1) - ((nq-1)/2)*np.log(n2) - ((nq*nl+1)/2)*np.log(n1*n2) - np.log(n1*n2))