import casadi as ca
from dual_ascent.dual_ascent import OptimizationProblemConfiguration
from admm.multi_block_admm import MultiBlockADMM
import numpy as np
import matplotlib  # 这样导入后，matplotlib.subplots 不存在
import matplotlib.pyplot as plt  # 然后使用 plt.subplots()


def plot_admm_convergence(admm_solver, result, figsize=(15, 16), save_path=None):
    """
    Plot the convergence and parameter changes during ADMM algorithm solving process

    Args:
        admm_solver: MultiBlockADMM instance
        result: Solution result dictionary
        figsize: Figure size
        save_path: Save path, if not None the image will be saved
    """
    history = result['convergence_history']
    iterations = len(history['primal_residuals'])
    iter_range = np.arange(iterations)

    # Create hyperparameter history records (this needs to be recorded in the solve method)
    if not hasattr(admm_solver, '_rho_history'):
        # If history not recorded, create an empty list
        rho_history = [admm_solver._rho] * iterations
        alpha_history = [admm_solver._alpha] * iterations
    else:
        rho_history = admm_solver._rho_history
        alpha_history = admm_solver._alpha_history

    # Create figure with 4 rows - 3 for plots and 1 for formula explanations
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 3, 3, 2])

    # 1. Primal residual and dual residual (log scale)
    ax1 = fig.add_subplot(gs[0])
    ax1.semilogy(iter_range, history['primal_residuals'], 'b-', linewidth=2, label='Primal Residual')
    ax1.semilogy(iter_range, history['dual_residuals'], 'r-', linewidth=2, label='Dual Residual')
    ax1.grid(True, which='both', linestyle='--', alpha=0.6)
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Residuals (log scale)')
    ax1.set_title('ADMM Convergence Process - Residual Changes')
    ax1.legend()

    # 2. Ratio of primal residual to dual residual
    ax2 = fig.add_subplot(gs[1])
    ratio = np.array(history['primal_residuals']) / (np.array(history['dual_residuals']) + 1e-10)
    ax2.plot(iter_range, ratio, 'g-', linewidth=2)
    ax2.axhline(y=10, color='r', linestyle='--', alpha=0.5, label=r'$r^k > 10 \cdot s^k$ (increase $\rho$)')
    ax2.axhline(y=0.1, color='b', linestyle='--', alpha=0.5, label=r'$s^k > 10 \cdot r^k$ (decrease $\rho$)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Residual Ratio r/s')
    ax2.set_title('Ratio of Primal to Dual Residuals')
    ax2.set_ylim([0, min(max(ratio), 20)])  # Set reasonable y-axis range
    ax2.legend()

    # 3. Hyperparameter changes
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(iter_range, rho_history[:iterations], 'b-', linewidth=2, label=r'Penalty parameter $\rho$')
    ax3.plot(iter_range, alpha_history[:iterations], 'r-', linewidth=2, label=r'Step size parameter $\alpha$')
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.set_xlabel('Iterations')
    ax3.set_ylabel('Parameter Value')
    ax3.set_title('ADMM Hyperparameter Changes')
    ax3.legend()

    # 4. Formula explanations in a dedicated text area at the bottom
    ax4 = fig.add_subplot(gs[3])
    ax4.axis('off')  # Hide axes

    formula_text = """
    Formula Explanations:

    1. Residuals:
       Primal Residual: $r^k = \sum_{i=1}^{N} \|A_i x_i^k - B_i\|_2$
       Dual Residual: $s^k = \\rho \sum_{i=1}^{N} \|A_i^T(x_i^k - x_i^{k-1})\|_2$

    2. Parameter Updates:
       $\\rho$ update: When $r^k > 10 \cdot s^k$, $\\rho \\leftarrow 1.5\\rho$; When $s^k > 10 \cdot r^k$, $\\rho \\leftarrow 0.8\\rho$
       $\\alpha$ update: Based on residual change trends, increase when converging, decrease when diverging

    3. Residual Ratio:
       Ratio $\\frac{r^k}{s^k}$ is used to adjust penalty parameter $\\rho$
    """

    ax4.text(0.5, 0.5, formula_text, ha='center', va='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='aliceblue', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    x1 = ca.SX.sym('x1', 1)
    x2 = ca.SX.sym('x2', 1)
    x3 = ca.SX.sym('x3', 1)
    x4 = ca.SX.sym('x4', 1)
    admm_solver = MultiBlockADMM()
    config = OptimizationProblemConfiguration(variables=[x1, x2],
                                              objective_function=x1 ** 2 + x2 ** 2)
    config2 = OptimizationProblemConfiguration(variables=[x3, x4],
                                              objective_function=x3 ** 2 + x4 ** 2)
    admm_solver.add_subproblem(config)
    admm_solver.add_subproblem(config2)
    admm_solver.set_linear_equality_constraint(ca.DM([[1,1,0,0],[0,0,1,-1]]),
                                               ca.DM([1,1]))

    admm_solver.generate_admm_functions()

    result = admm_solver.solve(tol=1e-6)
    print(result)
    plot_admm_convergence(admm_solver, result)
    pass
