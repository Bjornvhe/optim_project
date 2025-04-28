import numpy as np

def compute_gradient(alpha, G, y):
    """
    Compute the gradient of the dual objective function.
    grad = YGY alpha - 1
    """
    Y = np.diag(y)
    return Y @ (G @ (Y @ alpha)) - np.ones_like(alpha)

def compute_objective(alpha, G, y):
    """
    Compute the dual objective function value:
    f(alpha) = 1/2 * alpha^T * YGY * alpha - 1^T * alpha
    """
    Y = np.diag(y)
    return 0.5 * alpha.T @ Y @ G @ Y @ alpha - np.sum(alpha)

def exact_line_search(alpha, d, G, y):
    """
    Perform exact line search on the quadratic function.
    Solve for theta ∈ [0, 1] minimizing f(alpha + theta*d).
    """
    Y = np.diag(y)
    GY = G @ Y
    numerator = -np.dot(d, Y @ (GY @ (Y @ alpha)) - np.ones_like(alpha))
    denominator = np.dot(d, Y @ (GY @ (Y @ d)))

    if denominator <= 0:
        return 1.0  # fallback: just take full step if denominator not positive
    else:
        theta_star = numerator / denominator
        return np.clip(theta_star, 0, 1)  # restrict theta ∈ [0, 1]
