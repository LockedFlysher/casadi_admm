import casadi as ca
from dual_ascent.dual_ascent import OptimizationProblemConfiguration
from admm.multi_block_admm import MultiBlockADMM

# class OptimizationProblemConfiguration:
#     """优化问题配置类，用于存储优化问题的各种配置参数"""
#
#     def __init__(self, configuration: Dict[str, Any]):
#         """
#         初始化优化问题配置
#
#         Args:
#             configuration: 包含优化问题配置的字典，需包含以下键:
#                 - variables: 优化变量
#                 - objective_function: 目标函数
#                 - equality_constraints: 等式约束
#                 - inequality_constraints: 不等式约束
#                 - initial_guess: 初始猜测值
#         """
#         self.variables = configuration['variables']
#         self.objective_function = configuration['objective_function']
#         self.equality_constraints = configuration['equality_constraints']
#         self.inequality_constraints = configuration['inequality_constraints']
#         self.initial_guess = configuration['initial_guess']


if __name__ == '__main__':
    x1 = ca.SX.sym('x1',1)
    x2 = ca.SX.sym('x2',1)
    x3 = ca.SX.sym('x3',1)
    x4 = ca.SX.sym('x4',1)

    admm_solver = MultiBlockADMM()

    subproblem1 =  OptimizationProblemConfiguration(
        {"variables": [x1, x2],
         "objective_function": x1 ** 2 + x2 ** 2,
         "equality_constraints": {
             "A" : [ca.DM([1,1])],
             "B" : [ca.DM([1])]
         },
         "inequality_constraints": [],
         "initial_guess": [0, 0]
         })

    subproblem2 = OptimizationProblemConfiguration(
        {"variables": [x3, x4],
         "objective_function": x3 ** 2 + x4 ** 2,
         "equality_constraints": {
             "A" : [ca.DM([1,1])],
             "B" : [ca.DM([1])]
         },
         "inequality_constraints": [],
         "initial_guess": [0, 0]
         })
    admm_solver.add_subproblem(subproblem1)
    admm_solver.add_subproblem(subproblem2)
    admm_solver.generate_admm_functions()

    admm_solver.solve()

    pass