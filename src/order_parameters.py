import math
import numpy as np
from torch import nn
from itertools import product, combinations_with_replacement
from src.autoencoder import AutoEncoder
from src.data_manager import DataManager
from src.config import ExperimentConfig, DataManagerConfig, RunConfig
from src.integrals.integrals import Integral
from src.utils import set_seed
from scipy.integrate import solve_ivp

def compute_order_parameters_from_weights(model: AutoEncoder, omega):
    W = model.encoder.weight.data.detach().cpu().numpy()
    if model.tied: 
        V = model.encoder.weight.detach().cpu().numpy()
    else:
        V = model.decoder.weight.detach().cpu().numpy().T
    
    # W = np.zeros((model.encoder.weight.data.shape[0], model.encoder.weight.data.shape[1]))
    # V = np.zeros((model.decoder.weight.data.shape[1], model.decoder.weight.data.shape[0]))
    
    D, _ = omega.shape

    T1 = V @ omega @ V.T/(D**2)
    Q1 = W @ omega @ W.T/D
    R1 = V @ omega @ W.T/(D**1.5)
    
    print("q1")
    print(Q1)
    print("t1")
    print(T1)
    print("R1")
    print(R1)
    
    
    T0 = V@V.T/D
    Q0 = W@W.T/D
    R0 = W@V.T/D
    
    return T1, Q1, R1, T0, Q0, R0


def calculate_densities_from_weights(model: AutoEncoder, omega, psis, rhos):
    W = model.encoder.weight.data.detach().cpu().numpy()
    if model.tied: 
        V = model.encoder.weight.detach().cpu().numpy()
    else:
        V = model.decoder.weight.detach().cpu().numpy().T
    
    # W = np.zeros((model.encoder.weight.data.shape[0], model.encoder.weight.data.shape[1]))
    # V = np.zeros((model.decoder.weight.data.shape[1], model.decoder.weight.data.shape[0]))
    
    D, _ = omega.shape
    
    W_tau = W @ psis / math.sqrt(D) # K x D
    K = W_tau.shape[0]
    
    V_tau = V @ psis / math.sqrt(D) # K x D
    
    q, r, t = np.zeros((K,K,D)), np.zeros((K,K,D)), np.zeros((K,K,D))
    for k in range(K):
        for l in range(K):
            q[k,l] = W_tau[k] * W_tau[l]
            r[k,l] = V_tau[k] * W_tau[l] / math.sqrt(D)
            t[k,l] = V_tau[k] * V_tau[l] / D
    
    return q,r,t
    
    
def test_error_with_order_parameters(T0, T1, Q1, R1, omega, integral: Integral):
    ## taken from NonLinearShallowAE
    
    D, _ = omega.shape
    K, _ = Q1.shape
    
    C2Vec = np.zeros((K, K, K))
    C2Mat = np.zeros((K, K, K, K))
    
    for k in range(K):
        C2Vec[k, 0, 0] = T1[k, k]
        C2Vec[k, 0, 1] = R1[k, k]
        # C2Vec[k, 1, 0] = R1[k, k]
        C2Vec[k, 1, 1] = Q1[k, k]
        
        for l in range(K):
            C2Mat[k, l, 0, 0] = Q1[k, k]
            C2Mat[k, l, 0, 1] = Q1[k, l]
            # C2Mat[k, l, 1, 0] = Q1[k, l]
            C2Mat[k, l, 1, 1] = Q1[l, l]


    J2m = sum([integral.J2(C2Vec[k]) for k in range(K)])
    I2m = np.asarray([integral.I2(C2Mat[k,l])* T0[k,l] for k, l in product(range(K),range(K))])
    
    error  = 0.5*np.trace(omega)/D
    error += 0.5*np.sum(I2m)
    error -= np.sum(J2m)
    return error , (0.5*np.trace(omega)/D, np.sum(J2m), 0.5*np.sum(I2m) )

    
def change_in_T(q, r, t, rhos, T0, Q1, R1, T1, lr, K, D, integrals: Integral):
    
    
    def _computation(k, l):
        
        K, _ = T1.shape
        
        C2 = np.zeros((2, 2))
        C2[0,0] , C2[0,1] , C2[1,1] = Q1[k,k], Q1[k,k], Q1[k,k]
        
        dd = integrals.J2(C2) / Q1[k, k] * rhos * r[l, k]
        
        C2vec = np.zeros((K,2,2))
        for a in range(K):
            C2vec[a,0,0] , C2vec[a,0,1] , C2vec[a,1,1] = Q1[k,k], Q1[k,a], Q1[a,a]
        dd -= sum([integrals.I2(C2vec[a]) * t[a,l] * D for a in range(K)])
        
        return dd
    
    dkl = []
    dlk = []
    
    for k, l in combinations_with_replacement(range(K), 2):
        dkl.append(_computation(k, l))
        dlk.append(_computation(l, k))
        
    indicies = np.triu_indices_from(T0)
    
    dkl = lr * np.concatenate(dkl).reshape(len(indicies[0]), t.shape[-1])
    dlk = lr * np.concatenate(dlk).reshape(len(indicies[0]), t.shape[-1])
    
    dt = np.zeros(t.shape)
    dt[indicies] = dkl + dlk
    
    for k in range(K):
        for l in range(k):
            dt[k,l] = dt[l, k]
            
    return dt / D

    

def change_in_T0(q, r, t, rhos, T0, Q1, R1, T1, lr, K, D, integrals: Integral):
    
    def _computation(k, l):
        dd = 0
        K, _ = T1.shape
        
        C2 = np.zeros((2,2))
        C2[0,0] , C2[0,1] , C2[1,1] = Q1[k,k], Q1[k,k], Q1[k,k] #T1[l,l], R1[l,k], Q1[k,k]
        dd = integrals.J2(C2) / Q1[k,k] * R1[l,k]
        
        C2vec = np.zeros((K,2,2))
        for a in range(K):
            C2vec[a,0,0] , C2vec[a,0,1] , C2vec[a,1,1] = Q1[k,k], Q1[k,a], Q1[a,a]
        dd -= sum([integrals.I2(C2vec[a]) * T0[a,l] for a in range(K)])
        return dd
    
    indicies = np.triu_indices(K)
    
    dkl = [ _computation(k, l) for k, l in zip(*indicies)]
    dlk = [ _computation(l, k) for k, l in zip(*indicies)]
    
    dkl = lr * np.asarray(dkl)
    dlk = lr * np.asarray(dlk)

    dT0 = np.zeros_like(T0)
    dT0[indicies] = dkl + dlk
    for k in range(K):
        for l in range(k):
            dT0[k, l] = dT0[l, k]

    return dT0
    
    
def change_in_Q1(q, r, t, rhos, T0, Q1, R1, T1, lr, K, D, integrals: Integral):
    
    def _computation(k, l):
        C2 = np.zeros((2, 2))
        C2[0, 0] = Q1[k, k]
        C2[0, 1] = R1[k, k]
        C2[1, 1] = T1[k, k]

        det = Q1[k, k] * T1[k, k] - (R1[k, k] ** 2)
        C = integrals.I21(C2) * (T1[k, k] * q[k, l] - R1[k, k] * r[k, l])
        C += integrals.I22(C2) * (Q1[k, k] * r[k, l] - R1[k, k] * q[k, l])
        C /= det
        
        A = 0
        for a in range(K):
            if a == k:
                continue
            det_ka = Q1[k, k] * Q1[a, a] - (Q1[k, a] ** 2)

            C3 = np.zeros((3, 3))

            C3[0, 0] = Q1[k, k]; C3[0, 1] = Q1[k, k]
            C3[1, 0] = Q1[k, k]; C3[1, 1] = Q1[k, k]
            C3[0, 2] = Q1[k, a]; C3[2, 0] = Q1[k, a]
            C3[1, 2] = Q1[k, a]; C3[2, 1] = Q1[k, a]
            C3[2, 2] = Q1[a, a]
            Aa = integrals.I3(C3) * (Q1[a, a] * q[k, l] - Q1[k, a] * q[a, l])

            C3[0, 1] = Q1[k, a]; C3[1, 0] = Q1[k, a]
            C3[1, 1] = Q1[a, a]; C3[1, 2] = Q1[a, a]; C3[2, 1] = Q1[a, a]
            Aa += integrals.I3(C3) * (Q1[k, k] * q[a, l] - Q1[k, a] * q[k, l])

            Aa *= T0[k, a] / det_ka
            A -= Aa

        # ----- B term -----
        C3 = np.ones((3, 3)) * Q1[k, k]
        B = - T0[k, k] / Q1[k, k] * integrals.I3(C3) * q[k, l]

        return (A + B + C) * rhos

    indicies = np.triu_indices(K)
    dkl = [ _computation(k, l) for k, l in zip(*indicies) ]
    dlk = [ _computation(l, k) for k, l in zip(*indicies) ]

    dkl = lr * np.concatenate(dkl).reshape(len(indicies[0]), q.shape[-1])
    dlk = lr * np.concatenate(dlk).reshape(len(indicies[0]), q.shape[-1])

    dq = np.zeros_like(q)
    dq[indicies] = dkl + dlk
    for k in range(K):
        for l in range(k):
            dq[k, l] = dq[l, k]


    return dq / D
    

def change_in_R1(q, r, t, rhos, T0, Q1, R1, T1, lr, K, D, integrals: Integral):
    
    def _computation_dv(k, l):
        C2 = np.zeros((2, 2))
        C2[0, 0] = Q1[k, k]
        C2[0, 1] = Q1[k, k]
        C2[1, 0] = Q1[k, k]
        C2[1, 1] = Q1[k, k]

        dd = integrals.J2(C2) / Q1[k, k] * q[k, l] * rhos / D

        C2 = np.zeros((2, 2))
        for a in range(K):
            C2[0, 0] = Q1[k, k]
            C2[0, 1] = Q1[k, a]
            C2[1, 0] = Q1[k, a]
            C2[1, 1] = Q1[a, a]
            dd -= integrals.I2(C2) * r[a, l]
        return dd

    
    def _computation_dw(k, l):
        C2 = np.zeros((2, 2))
        C2[0, 0] = Q1[l, l]
        C2[0, 1] = R1[l, l]
        C2[1, 0] = R1[l, l]
        C2[1, 1] = T1[l, l]

        det = Q1[l, l] * T1[l, l] - (R1[l, l] ** 2)
        C = integrals.I21(C2) * (T1[l, l] * r[k, l] - R1[l, l] * t[k, l])
        C += integrals.I22(C2) * (Q1[l, l] * t[k, l] - R1[l, l] * r[k, l])
        C /= det

        # ----- A term -----
        A = 0
        for a in range(K):
            if a == l:
                continue
            det_la = Q1[l, l] * Q1[a, a] - (Q1[l, a] ** 2)

            C3 = np.zeros((3, 3))
            # first config
            C3[0, 0] = Q1[l, l]; C3[0, 1] = Q1[l, l]
            C3[1, 0] = Q1[l, l]; C3[1, 1] = Q1[l, l]
            C3[0, 2] = Q1[l, a]; C3[2, 0] = Q1[l, a]
            C3[1, 2] = Q1[l, a]; C3[2, 1] = Q1[l, a]
            C3[2, 2] = Q1[a, a]
            Aa = integrals.I3(C3) * (Q1[a, a] * r[k, l] - Q1[l, a] * r[k, a])

            # second config
            C3[0, 1] = Q1[l, a]; C3[1, 0] = Q1[l, a]
            C3[1, 1] = Q1[a, a]; C3[1, 2] = Q1[a, a]; C3[2, 1] = Q1[a, a]
            Aa += integrals.I3(C3) * (Q1[l, l] * r[k, a] - Q1[l, a] * r[k, l])

            Aa *= T0[l, a] / det_la
            A -= Aa

        # ----- B term -----
        C3 = np.ones((3, 3)) * Q1[l, l]
        B = - T0[l, l] / Q1[l, l] * integrals.I3(C3) * r[k, l]

        return (A + B + C) * rhos / D

    res_w = np.zeros_like(r)
    res_v = np.zeros_like(r)
    for k, l in product(range(K), range(K)):
        res_v[k, l] = _computation_dv(k, l)
        res_w[k, l] = _computation_dw(k, l)

    dr = lr * (res_w + res_v)
    return dr

    
def change_in_TBulk(Q1, T1_bulk, lr, K, integrals: Integral):
    
    def _computation(k, l):
        C2vec = np.zeros((K, 2, 2))
        for a in range(K):
            C2vec[a, 0, 0] = Q1[k, k]
            C2vec[a, 0, 1] = Q1[k, a]
            C2vec[a, 1, 1] = Q1[a, a]
        dd = 0.0
        for a in range(K):
            dd -= integrals.I2(C2vec[a]) * T1_bulk[a, l]
        return dd
    
    indices = np.triu_indices(K)
    dkl = [ _computation(k, l) for k, l in zip(*indices) ]
    dlk = [ _computation(l, k) for k, l in zip(*indices) ]

    dkl = lr * np.asarray(dkl)
    dlk = lr * np.asarray(dlk)

    dT = np.zeros_like(T1_bulk)
    dT[indices] = dkl + dlk
    for k in range(K):
        for l in range(k):
            dT[k, l] = dT[l, k]
    return dT


def change_in_RBulk(Q1,R1_bulk, lr, K, integrals: Integral):
    
    def _computation(k, l):
        C2vec = np.zeros((K, 2, 2))
        for a in range(K):
            C2vec[a, 0, 0] = Q1[k, k]
            C2vec[a, 0, 1] = Q1[k, a]
            C2vec[a, 1, 1] = Q1[a, a]
        dd = 0.0
        for a in range(K):
            dd -= integrals.I2(C2vec[a]) * R1_bulk[a, l]
        return dd

    dR = np.zeros_like(R1_bulk)
    for k, l in product(range(K), range(K)):
        dR[k, l] = lr * _computation(k, l)
        
    return dR

def set_size_dict(D: int, K: int, low_rank: int):
    return {"D": D, "K": K, "low_rank": low_rank, "bulk": (low_rank != D)}

def pack_order_params(state: dict[str, np.ndarray], sizes: dict[str, int]):
    K, L = sizes["K"], sizes["low_rank"]
    
    pack = [
        state["q"].reshape(-1),
        state["r"].reshape(-1),
        state["t"].reshape(-1),
        state["T0"].reshape(-1),
    ]
    
    if sizes["bulk"]:
        pack += [
            state["R1bulk"].reshape(-1),
            state["T1bulk"].reshape(-1)
        ]
        
    return np.concatenate(pack)
    
def unpack_order_params(state: dict[str, np.ndarray], sizes: dict[str, int]):
    K, L = sizes["K"], sizes["low_rank"]
    
    pt = 0
    
    n = K * K * L
    
    q = state[pt:pt+n].reshape(K, K, L)
    pt += n
    
    r = state[pt:pt+n].reshape(K, K, L)
    pt += n
    
    t = state[pt:pt+n].reshape(K, K, L)
    pt += n

    n = K * K
    T0 = state[pt:pt+n].reshape(K, K)
    pt += n
    
    
    R1bulk, T1bulk = None, None
    if sizes["bulk"]:
        R1bulk = state[pt:pt+n].reshape(K, K)
        pt += n
        T1bulk = state[pt:pt+n].reshape(K, K)
        pt += n
    
    
    return q, r, t, T0, R1bulk, T1bulk

import time

def solve_ODE(alpha, state, params):
    sizes = params["sizes"]

    D, K, low_rank = sizes["D"], sizes["K"], sizes["low_rank"]
    Q1Bulk = params["Q1Bulk"]
    rhos, Spikes = params["rhos"], params["Spikes"]
    integrals, lr = params["integrals"], params["lr"]
    
    bulk = low_rank != D
    
    q, r, t, T0, R1Bulk, T1Bulk = unpack_order_params(state, sizes)
    
    Q1_analytical, R1_analytical, T1_analytical = np.zeros((K, K)), np.zeros((K, K)), np.zeros((K, K))
    for k, l in product(range(K), range(K)):
        Q1_analytical[k, l] = (np.sum(Spikes * q[k, l]) / D)
        R1_analytical[k, l] = (np.sum(Spikes * r[k, l]) / D)
        T1_analytical[k, l] = (np.sum(Spikes * t[k, l]) / D)
        

    if bulk:
        Q1_analytical += Q1Bulk
        R1_analytical += R1Bulk
        T1_analytical += T1Bulk


    dR1Bulk, dT1Bulk = None, None
    if bulk:
        dt = change_in_T(q, r, t, Spikes, T0, Q1_analytical, R1_analytical, T1_analytical, lr, K, D, integrals)
        dq = change_in_Q1(q, r, t, Spikes, T0, Q1_analytical, R1_analytical, T1_analytical, lr, K, D, integrals)
        dr = change_in_R1(q, r, t, Spikes, T0, Q1_analytical, R1_analytical, T1_analytical, lr, K, D, integrals)
        
        dR1Bulk = change_in_RBulk(Q1_analytical, T1_analytical,  lr, K, integrals)
        dT1Bulk = change_in_TBulk(R1_analytical, T1_analytical,  lr, K, integrals)
    else:
        dt = change_in_T(q, r, t, rhos, T0, Q1_analytical, R1_analytical, T1_analytical, lr, K, D, integrals)
        dq = change_in_Q1(q, r, t, rhos, T0, Q1_analytical, R1_analytical, T1_analytical, lr, K, D, integrals)
        dr = change_in_R1(q, r, t, rhos, T0, Q1_analytical, R1_analytical, T1_analytical, lr, K, D, integrals)
            
    dT0 = change_in_T0(q, r, t, rhos, T0, Q1_analytical, R1_analytical, T1_analytical, lr, K, D, integrals)
    
    
    packed_order_params = pack_order_params({
        "q": dq,
        "r": dr,
        "t": dt,
        "T0": dT0,
        "R1bulk": dR1Bulk,
        "T1bulk": dT1Bulk
    }, sizes)
    
    return packed_order_params

def rk45(seed: int, config: RunConfig, low_rank: int, integral):
    
    set_seed(seed)
    
    dm_config: DataManagerConfig = config
    
    D = config.D
    K = config.K
    
    # load the model
    model = AutoEncoder(config=config)

    print(f"low rank: {low_rank}")

    # setup the data manager
    data_manager = DataManager(seed, config)

    (omega, sqrt_omega, eigenvalues, eigenvectors) = data_manager.get_trainloader()
    # testloader, (omega, sqrt_omega, eigenvalues, eigenvectors) = data_manager.get_testloader()
    
    psis = eigenvectors * np.sqrt(D)
    rhos = eigenvalues
    
    step = 0.
    dstep = 1.0/D
    
    # training the analytical integrals (if activation function has any)
    T1, Q1, R1, T0, Q0, R0 = compute_order_parameters_from_weights(model=model, omega=omega)

    Spikes = rhos[-low_rank:]
    Bulk = rhos[:-low_rank]

    Q1bulk = np.zeros(Q1.shape)
    R1bulk = np.zeros(R1.shape)
    T1bulk = np.zeros(T1.shape)

    q, r, t = calculate_densities_from_weights(model, omega, psis, rhos)
    for k, l in product(range(K), range(K)):
        Q1bulk[k,l] = np.sum(Bulk*q[k,l,:-low_rank]) / D
        R1bulk[k,l] = np.sum(Bulk*r[k,l,:-low_rank]) / D
        T1bulk[k,l] = np.sum(Bulk*t[k,l,:-low_rank]) / D

    q_analytical, r_analytical, t_analytical = (q.T[-low_rank:]).T, (r.T[-low_rank:]).T , (t.T[-low_rank:]).T
    Q1Bulk_analytical, R1Bulk_analytical, T1Bulk_analytical = Q1bulk, R1bulk, T1bulk
    T0_analytical = T0


    analytical_solutions_results = {
        "T": [],
        "T0": [],
        "Q1": [],
        "R1": [],
        "loss": []
    }

    Q1_analytical = np.zeros((K, K))
    R1_analytical = np.zeros((K, K))
    T1_analytical = np.zeros((K, K))
    
    max_step = 1000
    

    sizes = set_size_dict(D, K, low_rank)
    params = {
        "sizes": sizes,
        "Q1Bulk": Q1Bulk_analytical,
        "rhos": rhos,
        "Spikes": Spikes,
        "integrals": integral,
        "lr": config.learning_rate
    }
    state = {
        "q": q_analytical,
        "r": r_analytical,
        "t": t_analytical,
        "T0": T0_analytical,
        "R1bulk": R1Bulk_analytical,
        "T1bulk": T1Bulk_analytical
    }
    initial_state = pack_order_params(state, sizes)
    
    
    sol = solve_ivp(
        fun=lambda alpha, y: solve_ODE(alpha, y, params),
        t_span=(0, max_step),
        t_eval=None,
        y0=initial_state,
        method="RK45",
        rtol=1e-6,
        atol=1e-7,
        max_step=np.inf,
    )
    
    analytical_solutions_results = {"T": [], "T0": [], "Q1": [], "R1": [], "loss": []}
    print(sol)
    for col in range(sol.y.shape[1]):
        q, r, t, T0, R1Bulk, T1Bulk = unpack_order_params(sol.y[:, col], sizes)

        # Vectorized rebuild of order-parameter matrices
        Q1_analytical, R1_analytical, T1_analytical = np.zeros((K, K)), np.zeros((K, K)), np.zeros((K, K))
        for k, l in product(range(K), range(K)):
            Q1_analytical[k, l] = (np.sum(Spikes * q[k, l]) / D)
            R1_analytical[k, l] = (np.sum(Spikes * r[k, l]) / D)
            T1_analytical[k, l] = (np.sum(Spikes * t[k, l]) / D)
            
    
        if low_rank != D:
            Q1_analytical += Q1bulk
            R1_analytical += R1Bulk
            T1_analytical += T1Bulk

        loss, _ = test_error_with_order_parameters(T0, T1_analytical, Q1_analytical, R1_analytical, omega, integral)

            
        analytical_solutions_results["T0"].append(T0)
        analytical_solutions_results["T"].append(T1_analytical)
        analytical_solutions_results["Q1"].append(Q1_analytical)
        analytical_solutions_results["R1"].append(R1_analytical)
        analytical_solutions_results["loss"].append(loss)

    print(f"len of steps {len(analytical_solutions_results['loss'])}")
    
    C = np.zeros((8,8))
    C[:4, :4] = Q1_analytical
    C[:4, 4:] = R1_analytical
    C[4:, :4] = R1_analytical
    C[4:, 4:] = T1_analytical
    
    print(np.all(np.linalg.eigvals(C)) > -1e-9) 

    return sol.t, analytical_solutions_results["loss"]    

def euler(seed: int, config: RunConfig, low_rank: int, integral):
    set_seed(seed)
    
    dm_config: DataManagerConfig = config
    
    D = config.D
    K = config.K
    
    # load the model
    model = AutoEncoder(config=config)

    print(f"low rank: {low_rank}")

    # setup the data manager
    data_manager = DataManager(seed, config)

    # trainloader, (omega, sqrt_omega, eigenvalues, eigenvectors) = data_manager.get_trainloader()
    testloader, (omega, sqrt_omega, eigenvalues, eigenvectors) = data_manager.get_testloader()
    
    psis = eigenvectors * np.sqrt(D)
    rhos = eigenvalues
    
    step = 0.
    dstep = 1.0/D
    
    # training the analytical integrals (if activation function has any)
    T1, Q1, R1, T0, Q0, R0 = compute_order_parameters_from_weights(model=model, omega=omega)

    Spikes = rhos[-low_rank:]
    Bulk = rhos[:-low_rank]

    Q1bulk = np.zeros(Q1.shape)
    R1bulk = np.zeros(R1.shape)
    T1bulk = np.zeros(T1.shape)

    q, r, t = calculate_densities_from_weights(model, omega, psis, rhos)
    for k, l in product(range(K), range(K)):
        Q1bulk[k,l] = np.sum(Bulk*q[k,l,:-low_rank]) / D
        R1bulk[k,l] = np.sum(Bulk*r[k,l,:-low_rank]) / D
        T1bulk[k,l] = np.sum(Bulk*t[k,l,:-low_rank]) / D

    q_analytical, r_analytical, t_analytical = (q.T[-low_rank:]).T, (r.T[-low_rank:]).T , (t.T[-low_rank:]).T
    Q1Bulk_analytical, R1Bulk_analytical, T1Bulk_analytical = Q1bulk, R1bulk, T1bulk
    T0_analytical = T0
    
    analytical_solutions_results = {
        "T": [],
        "T0": [],
        "Q1": [],
        "R1": [],
        "loss": []
    }

    Q1_analytical = np.zeros((K, K))
    R1_analytical = np.zeros((K, K))
    T1_analytical = np.zeros((K, K))

    lr = config.learning_rate

    while step < config.alpha:
        
        for k, l in product(range(K), range(K)):
            Q1_analytical[k,l] = np.sum(Spikes*q_analytical[k,l]) / D
            R1_analytical[k,l] = np.sum(Spikes*r_analytical[k,l]) / D
            T1_analytical[k,l] = np.sum(Spikes*t_analytical[k,l]) / D
            
            # in a real dataset everything is treated as a spike
            if low_rank == D:
                Q1_analytical[k,l] += Q1Bulk_analytical[k,l]
                R1_analytical[k,l] += R1Bulk_analytical[k,l]
                T1_analytical[k,l] += T1Bulk_analytical[k,l]
        
        if low_rank == D: # means a real data set is being used        
            dt = change_in_T(q_analytical, r_analytical, t_analytical, rhos, T0_analytical, Q1_analytical, R1_analytical, T1_analytical, lr, K, D, integral)
            dq = change_in_Q1(q_analytical, r_analytical, t_analytical, rhos, T0_analytical, Q1_analytical, R1_analytical, T1_analytical, lr, K, D, integral)
            dr = change_in_R1(q_analytical, r_analytical, t_analytical, rhos, T0_analytical, Q1_analytical, R1_analytical, T1_analytical, lr, K, D, integral)
            dt0 = change_in_T0(q_analytical, r_analytical, t_analytical, rhos, T0_analytical, Q1_analytical, R1_analytical, T1_analytical, lr, K, D, integral)
        
        # NOTE: in the prior work at each step new order parameters are computed.
        # NOTE: as a change, we evolve them independently
        
        if low_rank == D:
        
            dt = change_in_T(q_analytical, r_analytical, t_analytical, rhos, T0_analytical, Q1_analytical, R1_analytical, T1_analytical, lr, K, D, integral)
            dq = change_in_Q1(q_analytical, r_analytical, t_analytical, rhos, T0_analytical, Q1_analytical, R1_analytical, T1_analytical, lr, K, D, integral)
            dr = change_in_R1(q_analytical, r_analytical, t_analytical, rhos, T0_analytical, Q1_analytical, R1_analytical, T1_analytical, lr, K, D, integral)
            dt0 = change_in_T0(q_analytical, r_analytical, t_analytical, rhos, T0_analytical, Q1_analytical, R1_analytical, T1_analytical, lr, K, D, integral)

        elif low_rank != D: 
                    
            dt = change_in_T(q_analytical, r_analytical, t_analytical, Spikes, T0_analytical, Q1_analytical, R1_analytical, T1_analytical, lr, K, D, integral)
            dq = change_in_Q1(q_analytical, r_analytical, t_analytical, Spikes, T0_analytical, Q1_analytical, R1_analytical, T1_analytical, lr, K, D, integral)
            dr = change_in_R1(q_analytical, r_analytical, t_analytical, Spikes, T0_analytical, Q1_analytical, R1_analytical, T1_analytical, lr, K, D, integral)
            dt0 = change_in_T0(q_analytical, r_analytical, t_analytical, Spikes, T0_analytical, Q1_analytical, R1_analytical, T1_analytical, lr, K, D, integral)
            
            # since in the synthetic dataset there are bulks, we also have to evolve them
            dR1Bulk_analytical = change_in_RBulk(Q1_analytical, R1Bulk_analytical, lr, K, integral)
            dT1Bulk_analytical = change_in_TBulk(Q1_analytical, T1Bulk_analytical, lr, K, integral)
                
        t_analytical += dt * dstep
        q_analytical += dq * dstep
        r_analytical += dr * dstep
        T0_analytical += dt0 * dstep 
        
        step += dstep
        
        if low_rank != D:
            R1Bulk_analytical += dR1Bulk_analytical * dstep
            T1Bulk_analytical += dT1Bulk_analytical * dstep
            
        
        loss, _ = test_error_with_order_parameters(T0_analytical, T1_analytical, Q1_analytical, R1_analytical, omega, integral)

        analytical_solutions_results["T0"].append(T0_analytical)
        analytical_solutions_results["T"].append(T1_analytical)
        analytical_solutions_results["Q1"].append(Q1_analytical)
        analytical_solutions_results["R1"].append(R1_analytical)
        analytical_solutions_results["loss"].append(loss)
        
        
    steps = np.arange(0, step, dstep)
    return steps, analytical_solutions_results["loss"]
        



def solve_autoencoder(seed: int, config: RunConfig, low_rank: int, integral):
    
    if config.ode_method == "rk45":
        return rk45(seed, config, low_rank, integral)
    
    elif config.ode_method == "euler":
        return euler(seed, config, low_rank, integral)
    
    else:
        return ValueError(f"ode_method in config is invalid: {config.ode_method}")
    
    