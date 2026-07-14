import numpy as np
import torch
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def generalized_combinations_torch(k):
    return k * (k - 1) / 2.0


def FSR_torch(X_temp, radius):
    if radius == 0:
        return (X_temp == X_temp.T).float()
    else:
        dist_matrix = torch.cdist(X_temp, X_temp, p=2)
        temp = 1.0 - dist_matrix
        temp[temp < radius] = 0
        return temp


def FCE_torch(FRSs=None, total_pairs=None, cardinalities=None):
    if cardinalities is None:
        cardinalities = torch.sum(FRSs, axis=1)
        
    indistinguishable_pairs = generalized_combinations_torch(cardinalities)
    
    if FRSs is not None and FRSs.ndim > 1:
        entropy_contributions = (total_pairs - indistinguishable_pairs) / total_pairs
        return torch.mean(entropy_contributions)
    else:
        return (total_pairs - indistinguishable_pairs) / total_pairs


def HFCE(X, sigma):
    X_torch = torch.from_numpy(X).float().to(device)
    n, m = X_torch.shape

    ID = (X >= 1).all(axis=0) & (X.max(axis=0) != X.min(axis=0))
    
    Imp = torch.zeros(m, device=device)
    FRSs = torch.zeros((m, n, n), device=device)
    
    total_pairs = n * (n - 1) / 2.0
    total_pairs_x = (n - 1) * (n - 2) / 2.0

    for k in range(m):
        if ID[k]:
            radius = 0
        else:
            radius = torch.std(X_torch[:, k]) / sigma
        
        temp = FSR_torch(X_torch[:, [k]], radius)
        FRSs[k] = temp
        
        fe = -torch.mean(torch.log2(torch.sum(temp, axis=1) / n + 1e-4))
        fce = -torch.mean(torch.log2(torch.sum(1 - temp, axis=1) / n + 1e-4))
        
        Imp[k] = fe + fce
        
    b_as = torch.argsort(Imp)
    
    weight_as = torch.zeros((n, m), device=device)
    weight_single = torch.zeros((n, m), device=device)
    FCE_as = torch.zeros(m, device=device)
    FCE_single = torch.zeros(m, device=device)
    FCE_as_x = torch.zeros((n, m), device=device)
    FCE_single_x = torch.zeros((n, m), device=device)
    rnc_as = torch.zeros((n, m), device=device)
    rnc_single = torch.zeros((n, m), device=device)
    
    for k in range(m):
        as_indices = b_as[:m - k]
        
        FRS_as = torch.stack([FRSs[idx] for idx in as_indices]).max(dim=0)[0] + 1e-4
        FRS_single = FRSs[k] + 1e-4

        weight_as[:, k] = torch.sqrt(torch.sum(FRS_as, axis=1) / n)
        weight_single[:, k] = torch.sqrt(torch.sum(FRS_single, axis=1) / n)
        
        FCE_as[k] = FCE_torch(FRSs=FRS_as, total_pairs=total_pairs)
        FCE_single[k] = FCE_torch(FRSs=FRS_single, total_pairs=total_pairs)

        sum_full_as = torch.sum(FRS_as)
        row_sums_as = torch.sum(FRS_as, axis=1)
        sum_after_delete_as = sum_full_as - 2 * row_sums_as + torch.diag(FRS_as)
        rnc_as[:, k] = row_sums_as - sum_after_delete_as / (n - 1)

        sum_full_single = torch.sum(FRS_single)
        row_sums_single = torch.sum(FRS_single, axis=1)
        sum_after_delete_single = sum_full_single - 2 * row_sums_single + torch.diag(FRS_single)
        rnc_single[:, k] = row_sums_single - sum_after_delete_single / (n - 1)
        
        fc_full_as = torch.sum(FRS_as, axis=1)
        fc_full_single = torch.sum(FRS_single, axis=1)
        for i in range(n):
            fc_sub_as = fc_full_as - FRS_as[:, i]
            fc_sub_deleted_as = torch.cat((fc_sub_as[:i], fc_sub_as[i+1:]))
            FCE_as_x[i, k] = torch.mean(FCE_torch(total_pairs=total_pairs_x, cardinalities=fc_sub_deleted_as))
            
            fc_sub_single = fc_full_single - FRS_single[:, i]
            fc_sub_deleted_single = torch.cat((fc_sub_single[:i], fc_sub_single[i+1:]))
            FCE_single_x[i, k] = torch.mean(FCE_torch(total_pairs=total_pairs_x, cardinalities=fc_sub_deleted_single))

    rne_x_as = 1 - FCE_as_x / FCE_as
    rne_x_as = torch.clamp(rne_x_as, 0, 1)
    
    nod_as_pos = rne_x_as * (n - torch.abs(rnc_as)) / (2 * n)
    nod_as_neg = rne_x_as * torch.sqrt((n + torch.abs(rnc_as)) / (2 * n))
    nod_as = torch.where(rnc_as > 0, nod_as_pos, nod_as_neg)
    
    rne_x_single = 1 - FCE_single_x / FCE_single
    rne_x_single = torch.clamp(rne_x_single, 0, 1)
    
    nod_single_pos = rne_x_single * (n - torch.abs(rnc_single)) / (2 * n)
    nod_single_neg = rne_x_single * torch.sqrt((n + torch.abs(rnc_single)) / (2 * n))
    nod_single = torch.where(rnc_single > 0, nod_single_pos, nod_single_neg)

    sum_single = torch.sum((1 - nod_single) * weight_single, axis=1)
    sum_as = torch.sum((1 - nod_as) * weight_as, axis=1)
    OS = 1 - (sum_single + sum_as) / (2 * m)
    
    return OS.cpu().numpy()