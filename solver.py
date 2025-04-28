import numpy as np
from projection import project_onto_feasible_set
from utils import compute_gradient, compute_objective, exact_line_search

class SVM_Solver:
    def __init__(self, G, y, C, tau_min=1e-5, tau_max=1e5, max_iter=1000, tol=1e-4, use_line_search=True):
        """
        Initialize the solver for dual SVM problem.

        Parameters:
        - G: Gram matrix (M x M)
        - y: labels vector (M,)
        - C: regularization parameter
        - tau_min, tau_max: step size bounds
        - max_iter: maximum number of iterations
        - tol: tolerance for stopping
        """
        self.use_line_search = use_line_search
        self.G = G
        self.y = y
        self.C = C
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.max_iter = max_iter
        self.tol = tol
        self.M = len(y)
        self.objective_values = []
        
    def solve(self, alpha_init=None):
        """
        Main solver using Projected Gradient Descent with step size adjustment and line search.
        """
        # Initialize alpha
        if alpha_init is None:
            alpha = np.zeros(self.M)
        else:
            alpha = alpha_init.copy()
        
        # Initialize step size
        tau = 1.0
        
        # Set up variables for Barzilai-Borwein step length
        s_prev = None
        z_prev = None
        
        # Line search control
        fbest = compute_objective(alpha, self.G, self.y)
        fref = np.inf
        fc = fbest
        L = 10
        ell = 0

        # Start iteration using for loop
        for k in range(self.max_iter):
            grad = compute_gradient(alpha, self.G, self.y)
            
            # Gradient step and projection
            beta = alpha - tau * grad # The gradient direction
            # To solve the dual problem, returns the nearest point to beta that satisfies the constraints
            alpha_proj = project_onto_feasible_set(beta, self.y, self.C) 
            d = alpha_proj - alpha 

            # Optional: do a line search if needed, added to see if negative stepsize was a problem
            if self.use_line_search and (k == 0 or compute_objective(alpha + d, self.G, self.y) > fref):
                theta = exact_line_search(alpha, d, self.G, self.y)
            else:
                theta = 1.0

            # Update alpha
            alpha_new = alpha + theta * d

            # Update reference function values
            f_new = compute_objective(alpha_new, self.G, self.y)
            self.objective_values.append(f_new)
            if f_new < fbest:
                fbest = f_new
                fc = f_new
                ell = 0
            else:
                fc = max(fc, f_new)
                ell += 1
                if ell == L:
                    fref = fc
                    fc = f_new
                    ell = 0

            # Check convergance criteria
            if np.linalg.norm(alpha_new - alpha) < self.tol:
                print(f"Converged at iteration {k}")
                break

            # Compute step length using Barzilai–Borwein method
            s = alpha_new - alpha
            z = compute_gradient(alpha_new, self.G, self.y) - grad

            if np.dot(s, z) > 0:
                tau_bb = np.dot(s, s) / np.dot(s, z)
                tau = max(min(tau_bb, self.tau_max), self.tau_min)
            else:
                tau = self.tau_max

            # Save for next iteration
            alpha = alpha_new.copy()
            s_prev, z_prev = s, z
            
            # Update on iterations during runtime
            if k % 100 == 0:
                print(f"Iteration {k}: Objective = {f_new:.6f}, Step size tau = {tau:.2e}")

        return alpha
