import pyomo.environ as pyo
from pyomo.gdp import Disjunct, Disjunction


def build_model():
    """
    Build the toy problem model

    Returns
    -------
    Pyomo.ConcreteModel
        Toy problem model
    """

    # Build Model
    m = pyo.ConcreteModel()

    # Sets
    m.set1 = pyo.RangeSet(1, 5, doc="set of first group of Boolean variables")
    m.set2 = pyo.RangeSet(1, 5, doc="set of second group of Boolean variables")

    m.sub1 = pyo.Set(initialize=[3], within=m.set1)

    # Variables
    m.Y2 = pyo.BooleanVar(m.set2, doc="Boolean variable associated to set 2")

    m.alpha = pyo.Var(
        within=pyo.Reals, bounds=(-0.1, 0.4), doc="continuous variable alpha"
    )
    m.beta = pyo.Var(
        within=pyo.Reals, bounds=(-0.9, -0.5), doc="continuous variable beta"
    )

    # Objective Function
    def obj_fun(m):
        """
        Objective function

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Toy problem model

        Returns
        -------
        Pyomo.Objective
            Build the objective function of the toy problem
        """
        return (
            4 * (pow(m.alpha, 2))
            - 2.1 * (pow(m.alpha, 4))
            + (1 / 3) * (pow(m.alpha, 6))
            + m.alpha * m.beta
            - 4 * (pow(m.beta, 2))
            + 4 * (pow(m.beta, 4))
        )

    m.obj = pyo.Objective(rule=obj_fun, sense=pyo.minimize, doc="Objective function")

    # First Disjunction
    def build_disjuncts1(m, set1):  # Disjuncts for first Boolean variable
        """
        Build disjuncts for the first Boolean variable

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Toy problem model
        set1 : RangeSet
            Set of first group of Boolean variables
        """

        def constraint1(m):
            """_summary_

            Parameters
            ----------
            m : Pyomo.ConcreteModel
                Toy problem model

            Returns
            -------
            Pyomo.Constraint
                Constraint that defines the value of alpha for each disjunct
            """
            return m.model().alpha == -0.1 + 0.1 * (
                set1 - 1
            )  # .model() is required when writing constraints inside disjuncts

        m.constraint1 = pyo.Constraint(rule=constraint1)

    m.Y1_disjunct = Disjunct(
        m.set1, rule=build_disjuncts1, doc="each disjunct is defined over set 1"
    )

    def Disjunction1(m):  # Disjunction for first Boolean variable
        """
        Disjunction for first Boolean variable

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Toy problem model

        Returns
        -------
        Pyomo.Disjunction
            Build the disjunction for the first Boolean variable set
        """
        return [m.Y1_disjunct[j] for j in m.set1]

    m.Disjunction1 = Disjunction(rule=Disjunction1, xor=False)

    # Second disjunction
    def build_disjuncts2(m, set2):  # Disjuncts for second Boolean variable
        """
        Build disjuncts for the second Boolean variable

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Toy problem model
        set2 : RangeSet
            Set of second group of Boolean variables
        """

        def constraint2(m):
            """_summary_

            Parameters
            ----------
            m : Pyomo.ConcreteModel
                Toy problem model

            Returns
            -------
            Pyomo.Constraint
                Constraint that defines the value of beta for each disjunct
            """
            return m.model().beta == -0.9 + 0.1 * (
                set2 - 1
            )  # .model() is required when writing constraints inside disjuncts

        m.constraint2 = pyo.Constraint(rule=constraint2)

    m.Y2_disjunct = Disjunct(
        m.set2, rule=build_disjuncts2, doc="each disjunct is defined over set 2"
    )

    def Disjunction2(m):  # Disjunction for first Boolean variable
        """
        Disjunction for second Boolean variable

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Toy problem model

        Returns
        -------
        Pyomo.Disjunction
            Build the disjunction for the second Boolean variable set
        """
        return [m.Y2_disjunct[j] for j in m.set2]

    m.Disjunction2 = Disjunction(rule=Disjunction2, xor=False)

    # # Associate boolean variables to disjuncts
    # for n2 in m.set2:
    #     m.Y2[n2].associate_binary_var(m.Y2_disjunct[n2].indicator_var)

    # Logical constraints

    # Constraint that allow to apply the reformulation over Y1
    def select_one_Y1(m):
        """
        Logical constraint that allows to apply the reformulation over Y1

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Toy problem model

        Returns
        -------
        Pyomo.LogicalConstraint
            Logical constraint that make Y1 to be true for only one element
        """
        return pyo.exactly(1, [m.Y1_disjunct[n].indicator_var for n in m.set1])

    m.oneY1 = pyo.LogicalConstraint(rule=select_one_Y1)

    # Constraint that allow to apply the reformulation over Y2
    def select_one_Y2(m):
        """
        Logical constraint that allows to apply the reformulation over Y2

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Toy problem model

        Returns
        -------
        Pyomo.LogicalConstraint
            Logical constraint that make Y2 to be true for only one element
        """
        return pyo.exactly(1, [m.Y2_disjunct[n].indicator_var for n in m.set2])

    m.oneY2 = pyo.LogicalConstraint(rule=select_one_Y2)

    # Constraint that define an infeasible region with respect to Boolean variables

    def infeasR_rule(m):
        """
        Logical constraint that defines an infeasible region with respect to Boolean variables

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Toy problem model

        Returns
        -------
        Pyomo.LogicalConstraint
            Logical constraint that defines an infeasible region on Y1[3]
        """
        return pyo.land([pyo.lnot(m.Y1_disjunct[j].indicator_var) for j in m.sub1])

    m.infeasR = pyo.LogicalConstraint(rule=infeasR_rule)

    return m


if __name__ == "__main__":
    # m = build_model()
    # pyo.TransformationFactory("gdp.bigm").apply_to(m)
    # solver = pyo.SolverFactory("gams")
    # solver.solve(m, solver="baron", tee=True)
    # print("Solution: alpha=", pyo.value(m.alpha), " beta=", pyo.value(m.beta))
    # print("Objective function value: ", pyo.value(m.obj))

    import time

    m = build_model()
    # begin time
    start_time = time.time()
    results=pyo.SolverFactory("gdpopt.ldbd").solve(
        m,
        starting_point=[5, 1],
        direction_norm="Linf",
        logical_constraint_list=[m.oneY1, m.oneY2],
        minlp_solver="gams",
        # minlp_solver_args=dict(solver="baron", add_options=["option optcr=0.001;"]),
        tee=True,
    )
    # end time
    end_time = time.time()
    print("Elapsed time: ", end_time - start_time)

    # Check the status and output results
    print("Solver Status: ", results.solver.status)
    print("Solver Termination Condition: ", results.solver.termination_condition)

    if results.solver.termination_condition == pyo.TerminationCondition.optimal:
        print("Solution: alpha = ", pyo.value(m.alpha))
        print("Solution: beta = ", pyo.value(m.beta))
        print("Objective value: ", pyo.value(m.obj))
    else:
        print("Solver did not find an optimal solution.")
    
