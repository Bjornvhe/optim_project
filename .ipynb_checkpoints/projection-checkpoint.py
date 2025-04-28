import numpy as np

def project_onto_feasible_set(beta, y, C, tol=1e-10, max_iter=100):
    """
    Project beta onto the feasible set:
      Omega = { alpha ∈ R^M : <y, alpha> = 0 and 0 <= alpha <= C }
      
    Uses bisection method to enforce the equality constraint.
    """

    def project_given_lambda(lambda_val):
        # Projection step: median between (0, C, beta + lambda * y)
        return np.clip(beta + lambda_val * y, 0, C)

    def inner_product(alpha):
        return np.dot(y, alpha)

    # Initialize lambda bounds
    lambda_min = -1e5
    lambda_max = 1e5

    # Find initial bracket
    alpha_min = project_given_lambda(lambda_min)
    alpha_max = project_given_lambda(lambda_max)
    
    if inner_product(alpha_min) > 0:
        while inner_product(alpha_min) > 0:
            lambda_min *= 2
            alpha_min = project_given_lambda(lambda_min)
    if inner_product(alpha_max) < 0:
        while inner_product(alpha_max) < 0:
            lambda_max *= 2
            alpha_max = project_given_lambda(lambda_max)

    # Bisection
    for _ in range(max_iter):
        lambda_mid = (lambda_min + lambda_max) / 2
        alpha_mid = project_given_lambda(lambda_mid)
        ip = inner_product(alpha_mid)

        if abs(ip) < tol:
            return alpha_mid

        if ip > 0:
            lambda_min = lambda_mid
        else:
            lambda_max = lambda_mid

    # If not converged, return the last midpoint
    return project_given_lambda(lambda_mid)
