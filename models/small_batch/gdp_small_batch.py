"""
gdp_small_batch.py
The gdp_small_batch.py module contains the GDP model for the small batch problem based on the Kocis and Grossmann (1988) paper.
The problem is based on the Example 4 of the paper.
The objective is to minimize the investment cost of the batch units.

References
----------
[1] Kocis, G. R.; Grossmann, I. E. Global Optimization of Nonconvex Mixed-Integer Nonlinear Programming (MINLP) Problems in Process Synthesis. Ind. Eng. Chem. Res. 1988, 27 (8), 1407-1421. https://doi.org/10.1021/ie00080a013

[2] Ovalle, D., Liñán, D. A., Lee, A., Gómez, J. M., Ricardez-Sandoval, L., Grossmann, I. E., & Bernal Neira, D. E. (2024). Logic-Based Discrete-Steepest Descent: A Solution Method for Process Synthesis Generalized Disjunctive Programs. arXiv preprint arXiv:2405.05358. https://doi.org/10.48550/arXiv.2405.05358

"""

import os
import pyomo.environ as pyo
from pyomo.core.base.misc import display
from pyomo.core.plugins.transform.logical_to_linear import (
    update_boolean_vars_from_binary,
)
from pyomo.gdp import Disjunct, Disjunction
from pyomo.opt.base.solvers import SolverFactory


def build_model():
    """
    Build the GDP model for the small batch problem.

    Returns
    -------
    m : Pyomo.ConcreteModel
        The GDP model for the small batch problem is created.

    References
    ----------
    [1] Kocis, G. R.; Grossmann, I. E. (1988). Global Optimization of Nonconvex Mixed-Integer Nonlinear Programming (MINLP) Problems in Process Synthesis. Ind. Eng. Chem. Res., 27(8), 1407-1421. https://doi.org/10.1021/ie00080a013

    [2] Ovalle, D., Liñán, D. A., Lee, A., Gómez, J. M., Ricardez-Sandoval, L., Grossmann, I. E., & Neira, D. E. B. (2024). Logic-Based Discrete-Steepest Descent: A Solution Method for Process Synthesis Generalized Disjunctive Programs. arXiv preprint arXiv:2405.05358. https://doi.org/10.48550/arXiv.2405.05358
    """
    NK = 3

    # Model
    m = pyo.ConcreteModel()

    # Sets
    m.i = pyo.Set(
        initialize=["a", "b"], doc="Set of products"
    )  # Set of products, i = a, b
    m.j = pyo.Set(
        initialize=["mixer", "reactor", "centrifuge"]
    )  # Set of stages, j = mixer, reactor, centrifuge
    m.k = pyo.RangeSet(
        NK, doc="Set of potential number of parallel units"
    )  # Set of potential number of parallel units, k = 1, 2, 3

    # Parameters and Scalars

    m.h = pyo.Param(
        initialize=6000, doc="Horizon time [hr]"
    )  # Horizon time  (available time) [hr]
    m.vlow = pyo.Param(
        initialize=250, doc="Lower bound for size of batch unit [L]"
    )  # Lower bound for size of batch unit [L]
    m.vupp = pyo.Param(
        initialize=2500, doc="Upper bound for size of batch unit [L]"
    )  # Upper bound for size of batch unit [L]

    # Demand of product i
    m.q = pyo.Param(
        m.i,
        initialize={"a": 200000, "b": 150000},
        doc="Production rate of the product [kg]",
    )
    # Cost coefficient for batch units
    m.alpha = pyo.Param(
        m.j,
        initialize={"mixer": 250, "reactor": 500, "centrifuge": 340},
        doc="Cost coefficient for batch units [$/L^beta*No. of units]]",
    )
    # Cost exponent for batch units
    m.beta = pyo.Param(
        m.j,
        initialize={"mixer": 0.6, "reactor": 0.6, "centrifuge": 0.6},
        doc="Cost exponent for batch units",
    )

    def coeff_init(m, k):
        """
        Coefficient for number of parallel units.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            The small batch GDP model.
        k : int
            The number of parallel units.

        Returns
        -------
        pyomo.Param
            Coefficient for number of parallel units. logarithm of k is applied to convexify the model.
        """
        return pyo.log(k)

    # Represent number of parallel units
    m.coeff = pyo.Param(
        m.k, initialize=coeff_init, doc="Coefficient for number of parallel units"
    )

    s_init = {
        ("a", "mixer"): 2,
        ("a", "reactor"): 3,
        ("a", "centrifuge"): 4,
        ("b", "mixer"): 4,
        ("b", "reactor"): 6,
        ("b", "centrifuge"): 3,
    }

    # Size factor for product i in stage j [kg/L]
    m.s = pyo.Param(
        m.i, m.j, initialize=s_init, doc="Size factor for product i in stage j [kg/L]"
    )

    t_init = {
        ("a", "mixer"): 8,
        ("a", "reactor"): 20,
        ("a", "centrifuge"): 4,
        ("b", "mixer"): 10,
        ("b", "reactor"): 12,
        ("b", "centrifuge"): 3,
    }

    # Processing time of product i in batch j [hr]
    m.t = pyo.Param(
        m.i, m.j, initialize=t_init, doc="Processing time of product i in batch j [hr]"
    )

    # Variables
    m.Y = pyo.BooleanVar(m.k, m.j, doc="Stage existence")  # Stage existence
    m.coeffval = pyo.Var(
        m.k,
        m.j,
        within=pyo.NonNegativeReals,
        bounds=(0, pyo.log(NK)),
        doc="Activation of Coefficient",
    )  # Activation of coeff
    m.v = pyo.Var(
        m.j,
        within=pyo.NonNegativeReals,
        bounds=(pyo.log(m.vlow), pyo.log(m.vupp)),
        doc="Colume of stage j [L]",
    )  # Volume of stage j [L]
    m.b = pyo.Var(
        m.i, within=pyo.NonNegativeReals, doc="Batch size of product i [L]"
    )  # Batch size of product i [L]
    m.tl = pyo.Var(
        m.i, within=pyo.NonNegativeReals, doc="Cycle time of product i [hr]"
    )  # Cycle time of product i [hr]
    # Number of units in parallel stage j
    m.n = pyo.Var(
        m.j, within=pyo.NonNegativeReals, doc="Number of units in parallel stage j"
    )

    # Constraints

    # Volume requirement in stage j
    @m.Constraint(m.i, m.j)
    def vol(m, i, j):
        """
        Volume Requirement for Stage j.
        Equation
        --------
        v_j \geq log(s_ij) + b_i for i = a, b and j = mixer, reactor, centrifuge

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The small batch GDP model.
        i : str
            Index of Product.
        j : str
            Stage.

        Returns
        -------
        Pyomo.Constraint
            A Pyomo constraint object representing the volume requirement for a given stage.
        """
        return m.v[j] >= pyo.log(m.s[i, j]) + m.b[i]

    # Cycle time for each product i
    @m.Constraint(m.i, m.j)
    def cycle(m, i, j):
        """
        Cycle time for each product i.

        Equation
        --------
        n_j + tl_i \geq log(t_ij) for i = a, b and j = mixer, reactor, centrifuge

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The small batch GDP model.
        i : str
            Index of Product.
        j : str
            Index of Stage.

        Returns
        -------
        Pyomo.Constraint
            A Pyomo constraint object representing the cycle time requirement for each product in each stage.
        """
        return m.n[j] + m.tl[i] >= pyo.log(m.t[i, j])

    # Constraint for production time
    @m.Constraint()
    def time(m):
        """
        Production time constraint.
        Equation:
            sum_{i \in I} q_i * \exp(tl_i - b_i) \leq h

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The small batch GDP model.

        Returns
        -------
        Pyomo.Constraint
            A Pyomo constraint object representing the production time constraint.
        """
        return sum(m.q[i] * pyo.exp(m.tl[i] - m.b[i]) for i in m.i) <= m.h

    # Relating number of units to 0-1 variables
    @m.Constraint(m.j)
    def units(m, j):
        """
        Relating number of units to 0-1 variables.
        Equation:
            n_j = sum_{k \in K} coeffval_{k,j} for j = mixer, reactor, centrifuge

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The small batch GDP model.
        j : str
            Index of Stage.

        Returns
        -------
        Pyomo.Constraint
            A Pyomo constraint object representing the relationship between the number of units and the binary variables.
        """
        return m.n[j] == sum(m.coeffval[k, j] for k in m.k)

    # Only one choice for parallel units is feasible
    @m.LogicalConstraint(m.j)
    def lim(m, j):
        """
        Only one choice for parallel units is feasible.
        Equation:
            sum_{k \in K} Y_{k,j} = 1 for j = mixer, reactor, centrifuge

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The small batch GDP model.
        j : str
            Index of Stage.

        Returns
        -------
        Pyomo.LogicalConstraint
            A Pyomo logical constraint ensuring only one choice for parallel units is feasible.
        """
        return pyo.exactly(1, m.Y[1, j], m.Y[2, j], m.Y[3, j])

    # _______ Disjunction_________

    def build_existence_equations(disjunct, k, j):
        """
        Build the Logic  Disjunct Constraints (equation) for the existence of the stage.

        Parameters
        ----------
        disjunct : Pyomo.Disjunct
            Disjunct block for the existence of the stage.
        k : int
            Number of parallel units.
        j : str
            Index of Stage.

        Returns
        -------
        None
            None, the constraints are built inside the disjunct.
        """
        m = disjunct.model()

        # Coefficient value activation
        @disjunct.Constraint()
        def coeffval_act(disjunct):
            """
            Coefficien value activation.

            Equation
            --------
            m.coeffval[k,j] = m.coeff[k] = log(k)

            Parameters
            ----------
            disjunct : Pyomo.Disjunct
                Disjunct block for the existence of the stage.

            Returns
            -------
            Pyomo.Constraint
                A Pyomo constraint object representing the activation of the coefficient value.
            """
            return m.coeffval[k, j] == m.coeff[k]

    def build_not_existence_equations(disjunct, k, j):
        """
        Build the Logic Disjunct Constraints (equations) for the absence of the stage.

        Parameters
        ----------
        disjunct : Pyomo.Disjunct
            Disjunct block for the absence of the stage.
        k : int
            Number of parallel units.
        j : str
            Index of Stage.

        Returns
        -------
        None
            None, the constraints are built inside the disjunct..
        """
        m = disjunct.model()

        # Coefficient value deactivation
        @disjunct.Constraint()
        def coeffval_deact(disjunct):
            """
            Coefficient value deactivation.

            Equation
            --------
            m.coeffval[k,j] = 0

            Parameters
            ----------
            disjunct : Pyomo.Disjunct
                Disjunct block for the absence of the stage.

            Returns
            -------
            Pyomo.Constraint
                A Pyomo constraint object representing the deactivation of the coefficient value.
            """
            return m.coeffval[k, j] == 0

    # Create disjunction block
    m.Y_exists = Disjunct(
        m.k, m.j, rule=build_existence_equations, doc="Existence of the stage"
    )
    m.Y_not_exists = Disjunct(
        m.k, m.j, rule=build_not_existence_equations, doc="Absence of the stage"
    )

    # Create disjunction

    @m.Disjunction(m.k, m.j)
    def Y_exists_or_not(m, k, j):
        """
        Build the Logical Disjunctions of the GDP model for the small batch problem.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The small batch GDP model.
        k : int
            Number of parallel units.
        j : str
            Index of Stage.

        Returns
        -------
        list
            List of disjuncts. The disjunction is between the existence and absence of the stage.
        """
        return [m.Y_exists[k, j], m.Y_not_exists[k, j]]

    
    # Link the global Boolean var Y to the disjunct's indicator_var using logic
    @m.LogicalConstraint(m.k, m.j)
    def link_Y_to_disjunction(m, k, j):
        # 意思是：当且仅当 Y 为真时，对应的析取项也为真
        return m.Y[k, j].equivalent_to(m.Y_exists[k, j].indicator_var)
    
    # # Associate Boolean variables with with disjunction
    # for k in m.k:
    #     for j in m.j:
    #         m.link_Y_to_disjunction(m, k, j)
    # ____________________________

    # Objective
    def obj_rule(m):
        """
        Objective: minimize the investment cost [$].

        Equation
        --------
        min z = sum(alpha[j] * exp(n[j] + beta[j]*v[j])) for j = mixer, reactor, centrifuge

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The small batch GDP model.

        Returns
        -------
        Pyomo.Objective
            Objective function to minimize the investment cost [$].
        """
        return sum(m.alpha[j] * (pyo.exp(m.n[j] + m.beta[j] * m.v[j])) for j in m.j)

    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    return m


if __name__ == "__main__":
    m = build_model()
    pyo.TransformationFactory("core.logical_to_linear").apply_to(m)
    pyo.TransformationFactory("gdp.bigm").apply_to(m)
    pyo.SolverFactory("gams").solve(
        m, solver="baron", tee=True, add_options=["option optcr=1e-6;"]
    )
    display(m)
    
    m = build_model()
    pyo.SolverFactory("gdpopt.ldbd").solve(
        m,
        subproblem_solver="gams",
        subproblem_solver_args=dict(
            solver="knitro",
            add_options=["option optcr=0.001;"],
            # tee=True,
            keepfiles=True,
        ),
        starting_point=[3, 3, 3],
        logical_constraint_list=[
            m.lim["mixer"],
            m.lim["reactor"],
            m.lim["centrifuge"],
        ],
        direction_norm="Linf",
        tee=True,
    )

    # from dsda_functions import solve_complete_external_enumeration
    
    # # 1. 定义逻辑函数：显式绑定 m.Y 和 Disjunct 的指示变量
    # def problem_logic_small_batch(m):
    #     logic_expr = []
    #     for k in m.k:
    #         for j in m.j:
    #             # 规则 1: 如果 m.Y[k,j] 为真，则 Y_exists[k,j] 激活
    #             logic_expr.append([m.Y[k, j], m.Y_exists[k, j].indicator_var])
                
    #             # 规则 2: 如果 m.Y[k,j] 为假 (即 ~m.Y)，则 Y_not_exists[k,j] 激活
    #             # 注意：fix_disjuncts 需要明确知道每个 Disjunct 是开还是关，所以两个都要设
    #             logic_expr.append([~m.Y[k, j], m.Y_not_exists[k, j].indicator_var])
    #     return logic_expr

    # # 2. 创建一个临时模型实例，仅用于获取变量和集合的引用
    # #    DSDA 函数内部会重新构建模型，但我们需要这些引用来告诉它“枚举哪个变量”
    # m_temp = build_model()

    # # 3. 定义外部变量字典 (External Variable Dictionary)
    # #    格式: {布尔变量对象: 它不仅要枚举的那个集合对象}
    # #    含义: 对于 m.Y[k, j]，我们在集合 m.k 上进行 "N选1" 的枚举
    # ext_ref = {m_temp.Y: m_temp.k}

    # # 4. 调用完全枚举求解函数
    # solve_complete_external_enumeration(
    #     model_function=build_model,  # 传入函数名，而不是模型对象
    #     model_args={},               # build_model 不需要参数
    #     ext_dict=ext_ref,            # 外部变量定义
    #     ext_logic=problem_logic_small_batch, # 逻辑定义
    #     mip_transformation=False,    # 是否在内部转化为 MIP (设为 False 使用 GDP BigM)
    #     transformation='bigm',       # 使用 BigM 变换
    #     feasible_model='small_batch',# 用于生成结果文件名的标签
    #     subproblem_solver='gams',    # 子问题求解器接口
    #     subproblem_solver_options=dict(
    #         solver='baron',          # 具体求解器 (或 dicopt/knitro)
    #         add_options=["option optcr=0.001;"],
    #         keepfiles=True
    #     ),
    #     iter_timelimit=100,          # 每个子问题的时间限制
    #     timelimit=3600,              # 总时间限制
    #     gams_output=False,
    #     # tee=True,                    # 显示子问题求解日志
    #     global_tee=True,             # 显示 DSDA 主算法日志
    #     export_csv=True              # 将结果导出为 CSV
    # )
