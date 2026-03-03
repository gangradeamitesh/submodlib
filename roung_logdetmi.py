import torch

def logdet_psd(M, eps=1e-6):
    """
    Computes log(det(M)) for PSD matrix using Cholesky
    """
    L = torch.linalg.cholesky(M + eps * torch.eye(M.shape[0], device=M.device))
    return 2 * torch.sum(torch.log(torch.diagonal(L)))

def logdet_mutual_information(
    S_A,
    S_AQ,
    S_Q_inv,
    eta
):
    """
    Computes:
    log det(S_A) - log det(S_A - eta^2 S_AQ S_Q^{-1} S_AQ^T)
    """
    term1 = logdet_psd(S_A)

    correction = eta**2 * S_AQ @ S_Q_inv @ S_AQ.T
    term2 = logdet_psd(S_A - correction)

    return term1 - term2

def marginal_gain(
    A,
    i,
    S_data,
    S_data_query,
    S_Q_inv,
    eta,
    lambdaVal=1e-6
):
    """
    Computes Δ(i | A)
    """
    idx = A + [i]

    S_A = S_data[idx][:, idx] + lambdaVal * torch.eye(len(idx))
    S_AQ = S_data_query[idx]

    val_new = logdet_mutual_information(S_A, S_AQ, S_Q_inv, eta)

    if len(A) == 0:
        return val_new

    S_A_old = S_data[A][:, A] + lambdaVal * torch.eye(len(A))
    S_AQ_old = S_data_query[A]

    val_old = logdet_mutual_information(S_A_old, S_AQ_old, S_Q_inv, eta)

    return val_new - val_old

def greedy_ldmi(
    S_data,
    S_data_query,
    S_query,
    k,
    eta=1.0,
    lambdaVal=1e-6
):
    """
    Greedy maximization of LDMI
    """
    n = S_data.shape[0]
    device = S_data.device

    S_Q_inv = torch.linalg.inv(
        S_query + lambdaVal * torch.eye(S_query.shape[0], device=device)
    )

    A = []
    remaining = set(range(n))

    for step in range(k):
        best_gain = -float("inf")
        best_elem = None

        for i in remaining:
            gain = marginal_gain(
                A,
                i,
                S_data,
                S_data_query,
                S_Q_inv,
                eta,
                lambdaVal
            )
            if gain > best_gain:
                best_gain = gain
                best_elem = i

        A.append(best_elem)
        remaining.remove(best_elem)

        print(f"Step {step+1}: selected {best_elem}, gain={best_gain:.4f}")

    return A
n = 50
m = 5

torch.manual_seed(0)

# Fake similarity matrices (PSD)
X = torch.randn(n, 16)
Q = torch.randn(m, 16)

S_data = X @ X.T
S_query = Q @ Q.T
S_data_query = X @ Q.T

selected = greedy_ldmi(
    S_data,
    S_data_query,
    S_query,
    k=10,
    eta=2.0
)

print("Selected indices:", selected)
