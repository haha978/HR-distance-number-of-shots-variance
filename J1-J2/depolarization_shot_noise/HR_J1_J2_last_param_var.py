import qiskit
from qiskit import QuantumCircuit, Aer
from qiskit.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
from azure.quantum.qiskit import AzureQuantumProvider
from qiskit import transpile
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
import os
from Circuit import Q_Circuit
from utils import expectation_X, get_NN_coupling, get_nNN_coupling, get_exp_cross
from utils import flatten_neighbor_l, get_nearest_neighbors, get_next_nearest_neighbors
from utils import distanceVecFromSubspace

HR_dist_hist = []

def get_args(parser):
    parser.add_argument('--input_dir', type = str, help = "directory where VQE_hyperparam_dict.npy exists and HR distances and plots will be stored")
    parser.add_argument('--shots', type=int, default=1000, help = "number of shots during HamiltonianReconstuction (default: 1000)")
    parser.add_argument('--num_HR', type = int, help = "number of HR distance measurements to get the variance/mean")
    args = parser.parse_args()
    return args

def get_measurement(n_qbts, var_params, backend, h_l, hyperparam_dict, param_idx):
    num_shots = hyperparam_dict["shots"]
    backendnm = hyperparam_dict["backend"]
    measurement_path = os.path.join(args.input_dir, "measurement", f"{param_idx}th_param_{''.join([str(e) for e in h_l])}qbt_h_gate.npy")
    m, n = hyperparam_dict["m"], hyperparam_dict["n"]
    circ = Q_Circuit(m, n, var_params, h_l, hyperparam_dict["n_layers"], hyperparam_dict["ansatz_type"])
    circ.measure(list(range(n_qbts)), list(range(n_qbts)))
    circ = transpile(circ, backend)
    job = backend.run(circ, shots = num_shots)
    if backendnm != "aer_simulator":
        job_id = job.id()
        job_monitor(job)
    result = job.result()
    measurement = dict(result.get_counts())
    np.save(measurement_path, measurement)
    return measurement

def get_params(params_dir_path, param_idx):
    var_params = np.load(os.path.join(params_dir_path, f"var_params_{param_idx}.npy"))
    return var_params

def get_measurement_index_l(h_idx, z_indices):
    m_index_l = []
    for zi, zj in z_indices:
        if h_idx != zi and h_idx != zj:
            m_index_l.append([h_idx, zi, zj])
    return m_index_l

def get_HR_distance(hyperparam_dict, param_idx, params_dir_path, backend):
    cov_mat = np.zeros((3,3))
    m, n = hyperparam_dict["m"],  hyperparam_dict["n"]
    n_qbts = m * n
    z_l, x_l = [], [i for i in range(n_qbts)]
    var_params = get_params(params_dir_path, param_idx)
    z_m = get_measurement(n_qbts, var_params, backend, z_l, hyperparam_dict, param_idx)
    x_m = get_measurement(n_qbts, var_params, backend, x_l, hyperparam_dict, param_idx)
    exp_X, exp_NN, exp_nNN = expectation_X(x_m, 1), get_NN_coupling(z_m, m, n, 1), get_nNN_coupling(z_m, m, n, 1)

    #diagonal terms
    cov_mat[0, 0] = expectation_X(x_m, 2) - exp_X**2
    cov_mat[1, 1] = get_NN_coupling(z_m, m, n, 2) - exp_NN**2
    cov_mat[2, 2] = get_nNN_coupling(z_m, m, n, 2) - exp_nNN**2

    #cross terms
    NN_index_l = flatten_neighbor_l(get_nearest_neighbors(m, n), m, n)
    nNN_index_l = flatten_neighbor_l(get_next_nearest_neighbors(m, n), m, n)
    NN_nNN_val = - (exp_NN * exp_nNN)

    for NN_indices in NN_index_l:
        for nNN_indices in nNN_index_l:
            indices = NN_indices + nNN_indices
            NN_nNN_val += get_exp_cross(z_m, indices)

    cov_mat[1, 2], cov_mat[2, 1]= NN_nNN_val, NN_nNN_val
    X_NN_val = -(exp_X * exp_NN)
    X_nNN_val = -(exp_X * exp_nNN)

    for h_idx in range(n_qbts):
        h_l = [h_idx]
        cross_m = get_measurement(n_qbts, var_params, backend, h_l, hyperparam_dict, param_idx)
        X_NN_index_l = get_measurement_index_l(h_idx, NN_index_l)
        X_nNN_index_l = get_measurement_index_l(h_idx, nNN_index_l)
        for indices in X_NN_index_l:
            X_NN_val += get_exp_cross(cross_m, indices)
        for indices in X_nNN_index_l:
            X_nNN_val += get_exp_cross(cross_m, indices)
    cov_mat[0, 1] = X_NN_val
    cov_mat[0, 2] = X_nNN_val
    cov_mat[2, 0], cov_mat[1, 0] = cov_mat[0, 2], cov_mat[0, 1]
    val, vec = np.linalg.eigh(cov_mat)
    argsort = np.argsort(val)
    val, vec = val[argsort], vec[:, argsort]
    orig_H = np.array([1, hyperparam_dict["J1"], hyperparam_dict["J2"]])
    orig_H = orig_H/np.linalg.norm(orig_H)
    HR_dist = distanceVecFromSubspace(orig_H, vec[:, :1])
    return HR_dist

def main(args):
    if not os.path.exists(os.path.join(args.input_dir,"VQE_hyperparam_dict.npy")):
        raise ValueError( "input directory must be a valid input path that contains VQE_hyperparam_dict.npy")
    if not os.path.isdir(os.path.join(args.input_dir, "measurement")):
        os.makedirs(os.path.join(args.input_dir, "measurement"))

    #LOAD All the hyperparamter data from VQE here
    VQE_hyperparam_dict = np.load(os.path.join(args.input_dir, "VQE_hyperparam_dict.npy"), allow_pickle = True).item()
    params_dir_path = os.path.join(args.input_dir,"params_dir")
    backend_name = "aer_simulator"
    backend = Aer.get_backend(backend_name)

    hyperparam_dict = {}
    hyperparam_dict["gst_E"] = VQE_hyperparam_dict["gst_E"]
    hyperparam_dict["J1"], hyperparam_dict["J2"] = VQE_hyperparam_dict["J1"], VQE_hyperparam_dict["J2"]
    hyperparam_dict["m"], hyperparam_dict["n"] = VQE_hyperparam_dict["m"], VQE_hyperparam_dict["n"]
    hyperparam_dict["n_layers"] = VQE_hyperparam_dict["n_layers"]
    hyperparam_dict["ansatz_type"] = VQE_hyperparam_dict["ansatz_type"]

    #Need a new number of shots for HR distance --> args.shots set when running the script
    hyperparam_dict["shots"] = args.shots
    hyperparam_dict["backend"] = backend_name

    #save hyperparamter dictionary as "HR_hyperparam_dict.npy"
    print("This is hyperparameter dictionary newly constructed: ", hyperparam_dict)
    np.save(os.path.join(args.input_dir, "HR_hyperparam_dict.npy"), hyperparam_dict)

    #load simulated energy --> need this to get the last parameters' index
    with open(os.path.join(args.input_dir, "E_hist.pkl"), "rb") as fp:
        E_hist = pickle.load(fp)

    gst_E = hyperparam_dict["gst_E"]
    m, n = hyperparam_dict["m"], hyperparam_dict["n"]
    J1, J2 = hyperparam_dict["J1"], hyperparam_dict["J2"]
    n_layers = hyperparam_dict["n_layers"]

    HR_dist_l = []
    fid_hist = []
    last_param_idx = len(E_hist) - 1
    for ith_HR_dist in list(range(args.num_HR)):
        HR_dist = get_HR_distance(hyperparam_dict, last_param_idx, params_dir_path, backend)
        HR_dist_l.append(HR_dist)
    print(HR_dist_l)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "VQE for 2-D J1-J2 model")
    args = get_args(parser)
    main(args)
