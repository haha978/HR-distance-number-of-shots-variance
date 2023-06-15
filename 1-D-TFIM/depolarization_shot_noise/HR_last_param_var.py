import qiskit
from qiskit import QuantumCircuit, Aer
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
from azure.quantum.qiskit import AzureQuantumProvider
from qiskit import transpile
import numpy as np
import argparse
from utils import distanceVecFromSubspace, get_exp_cross, get_exp_X, get_exp_ZZ
import pickle
import matplotlib.pyplot as plt
import os

def get_args(parser):
    parser.add_argument('--input_dir', type = str, help = "directory where hyperparam_dict.npy exists and HR distances and plots will be stored")
    parser.add_argument('--shots', type=int, default=1000, help = "number of shots used during HamiltonianReconstuction (default: 1000)")
    parser.add_argument('--num_HR', type = int, help = "number of HR distance measurements to get the variance/mean")
    parser.add_argument('--p1', type = float, default = 0.0065, help = "one-qubit gate depolarization noise (default: 0.0065)")
    parser.add_argument('--p2', type = float, default = 0.0398, help = "two-qubit gate depolarization noise (default: 0.0398)")
    args = parser.parse_args()
    return args

def Q_Circuit(N_qubits, var_params, h_l, n_layers):
    circ = QuantumCircuit(N_qubits, N_qubits)
    param_idx = 0
    for i in range(N_qubits):
        circ.h(i)
    if N_qubits % 2 == 0:
        for layer in range(n_layers):
            if layer % 2 == 0:
                for i in range(0, N_qubits, 2):
                    circ.cx(i, i+1)
                for i in range(N_qubits):
                    circ.ry(var_params[param_idx], i)
                    param_idx += 1
            else:
                for i in range(1, N_qubits-1, 2):
                    circ.cx(i, i+1)
                for i in range(1, N_qubits-1):
                    circ.ry(var_params[param_idx], i)
                    param_idx += 1
    else:
        for layer in range(n_layers):
            if layer % 2 == 0:
                for i in range(0, N_qubits-1, 2):
                    circ.cx(i, i+1)
                for i in range(N_qubits-1):
                    circ.ry(var_params[param_idx], i)
                    param_idx += 1
            else:
                for i in range(1, N_qubits, 2):
                    circ.cx(i, i+1)
                for i in range(1, N_qubits):
                    circ.ry(var_params[param_idx], i)
                    param_idx += 1
    for h_idx in h_l:
        circ.h(h_idx)
    return circ

def get_measurement(n_qbts, var_params, backend, h_l, hyperparam_dict, ith_HR_dist):
    shots = hyperparam_dict["shots"]
    measurement_path = os.path.join(args.input_dir, "measurement", f"{ith_HR_dist}th_{shots}_shots_{''.join([str(e) for e in h_l])}qbt_h_gate.npy")
    if os.path.exists(measurement_path):
        #no need to save as it is already saved
        measurement = np.load(measurement_path, allow_pickle = "True").item()
    else:
        circ = Q_Circuit(n_qbts, var_params, h_l, hyperparam_dict["n_layers"])
        circ.measure(list(range(n_qbts)), list(range(n_qbts)))
        circ = transpile(circ, backend)
        job = backend.run(circ, shots = shots)
        # only using simulator
        # if hyperparam_dict["backend"] != "aer_simulator":
        #     job_id = job.id()
        #     job_monitor(job)
        result = job.result()
        measurement = dict(result.get_counts())
        np.save(measurement_path, measurement)
    return measurement

def get_HR_distance(hyperparam_dict, var_params, backend, ith_HR_dist):
    cov_mat = np.zeros((2,2))
    n_qbts = hyperparam_dict["n_qbts"]
    #need to delete the below as well
    z_l, x_l = [], [i for i in range(n_qbts)]
    z_m = get_measurement(n_qbts, var_params, backend, z_l, hyperparam_dict, ith_HR_dist)
    x_m = get_measurement(n_qbts, var_params, backend, x_l, hyperparam_dict, ith_HR_dist)
    exp_X, exp_ZZ = get_exp_X(x_m, 1),  get_exp_ZZ(z_m, 1)
    cov_mat[0, 0] =  get_exp_X(x_m, 2) - exp_X**2
    cov_mat[1, 1] = get_exp_ZZ(z_m, 2) - exp_ZZ**2
    cross_val = 0
    z_indices = [[i, i+1] for i in range(n_qbts) if i != (n_qbts-1)]
    for h_idx in range(n_qbts):
        h_l = [h_idx]
        cross_m = get_measurement(n_qbts, var_params, backend, h_l, hyperparam_dict, ith_HR_dist)
        for z_ind in z_indices:
            if h_idx not in z_ind:
                indices = h_l + z_ind
                cross_val += get_exp_cross(cross_m, indices)
    cov_mat[0,1] = cross_val - exp_X*exp_ZZ
    cov_mat[1,0] = cov_mat[0,1]
    val, vec = np.linalg.eigh(cov_mat)
    argsort = np.argsort(val)
    val, vec = val[argsort], vec[:, argsort]
    orig_H = np.array([1, hyperparam_dict["J"]])
    orig_H = orig_H/np.linalg.norm(orig_H)
    HR_dist = distanceVecFromSubspace(orig_H, vec[:, :1])
    return HR_dist

def get_last_param(input_dir):
    params_dir_path = os.path.join(input_dir, "params_dir")
    param_fn_l = [ file for file in os.listdir(params_dir_path) if file[:10] == "var_params" ]
    last_param_fn = param_fn_l[0]
    for param_fn in param_fn_l:
        param_idx = int(param_fn[11:-4])
        if param_idx > int(last_param_fn[11:-4]):
            last_param_fn = param_fn
    print(last_param_fn)
    return np.load(os.path.join(params_dir_path, last_param_fn))

def get_hyperparam_dict(input_dir, shots, p1, p2):
    """
    Returns hyperparameter dictionary that contains following keys and values

    n_qbts: number of qubits
    J: coupling strength
    shots: number of shots used to get HR distance
    n_layers: number of layers`
    gst_E: ground state energy of system of interest
    """
    hyperparam_dict = {}
    VQE_dict = np.load(os.path.join(input_dir,"VQE_hyperparam_dict.npy"), allow_pickle = True).item()
    hyperparam_dict['n_qbts'], hyperparam_dict['J'] = VQE_dict['n_qbts'], VQE_dict['J']
    hyperparam_dict['n_layers'], hyperparam_dict['gst_E'] = VQE_dict['n_layers'], VQE_dict['gst_E']
    hyperparam_dict['shots'] = shots
    hyperparam_dict["p1"], hyperparam_dict["p2"] = args.p1, args.p2
    print(hyperparam_dict)
    np.save(os.path.join(input_dir, "HR_hyperparam_dict.npy"), hyperparam_dict)
    return hyperparam_dict

def main(args):
    #check whether args.input_dir has appropriate files and folders. If not throw error or create needed files
    assert os.path.exists(os.path.join(args.input_dir,"VQE_hyperparam_dict.npy")), "input directory must be a valid input path that contains hyperparam_dict.npy"
    assert os.path.exists(os.path.join(args.input_dir,"params_dir")), "input directory must be a valid input path that contains params directory"
    if not os.path.isdir(os.path.join(args.input_dir, "measurement")):
        os.makedirs(os.path.join(args.input_dir, "measurement"))

    #load the last parameter in the VQE iterations
    var_params = get_last_param(args.input_dir)

    #get hyperparameter dictionary to measure HR distance. get_hyperparam_dict is also where the number of
    hyperparam_dict = get_hyperparam_dict(args.input_dir, args.shots, args.p1, args.p2)

    #for now use the simulator, can change the backend to actual hardware or ionq simulator if needed in the future
    noise_model = NoiseModel()
    p1_error = depolarizing_error(args.p1, 1)
    p2_error = depolarizing_error(args.p2, 2)
    noise_model.add_all_qubit_quantum_error(p1_error, ['h', 'ry'])
    noise_model.add_all_qubit_quantum_error(p2_error, ['cx'])
    backend = AerSimulator(noise_model = noise_model)

    HR_dist_l = []
    for ith_HR_dist in list(range(args.num_HR)):
        HR_dist = get_HR_distance(hyperparam_dict, var_params, backend, ith_HR_dist)
        HR_dist_l.append(HR_dist)

    #list of HR distance measurements
    HR_dist_l = np.array(HR_dist_l)
    np.save(os.path.join(args.input_dir, f"p1_{args.p1}_p2_{args.p2}_{args.num_HR}_HR_dists__{args.shots}_shots.npy"), HR_dist_l)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "HR distance variance measurement")
    args = get_args(parser)
    main(args)
