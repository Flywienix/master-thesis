import casadi


def setup_MPC(N, model, price, sol, Tamb, X0):
    # ------- MPC parameters -------
    # N: prediction horizon (normally something like 1 day = 96 time steps when assuming 15 min steps).
    u_max = 1  # maximum heating power
    u_min = 0  # minimum heating power
    y_max = 22  # maximum indoor temperature
    y_min = 20  # minimum indoor temperature

    # create optimization model
    m = casadi.Opti()

    # state space model
    A = model["A"]
    B = model["B"]
    C = model["C"]

    # optimization variable
    x = m.variable(5, N + 1)
    y = m.variable(1, N)
    u = m.variable(3, N)
    slack_min = m.variable(1, N)
    slack_max = m.variable(1, N)

    obj = 0
    for k in range(N):
        # obj += price[k][0] * u[0, k] + 1e6 * (slack_min[0, k]*slack_min[0,k] + slack_max[0, k]*slack_max[0,k])
        obj += price[k] * u[0, k] + 1e6 * (
            slack_min[0, k] * slack_min[0, k] + slack_max[0, k] * slack_max[0, k]
        )

    # set the objective
    m.minimize(obj)

    for k in range(N):
        # dynamics
        m.subject_to(x[:, k + 1] == A @ x[:, k] + B @ u[:, k])
        m.subject_to(y[k] == C @ x[:, k])
        # we also have to set the inputs of the ambient air temperature and solar radiation forecasts to their respective values
        # m.subject_to(u[1, k] == sol[k][0])
        # m.subject_to(u[2, k] == Tamb[k][0])
        m.subject_to(u[1, k] == sol[k])
        m.subject_to(u[2, k] == Tamb[k])
        # this leaves the input u[0] (i.e., the heating input) as the only optimization variable left that the controller
        # has to decide

    # we have to set the initial condition again (just as for the simulation)
    m.subject_to(x[:, 0] == X0)

    # here we need "slack variables" because the mpc cannot guarantee that the constraints are satisfied exactly
    # and we have to ensure feasibility
    # add constraints with slack variables
    m.subject_to(m.bounded(y_min - slack_min, y[:], y_max + slack_max))
    m.subject_to(slack_min >= 0)
    m.subject_to(slack_max >= 0)
    # the heating intensity is limited between 0 and 1
    m.subject_to(m.bounded(u_min, u[0, :], u_max))
    return m, y, x, u


def solve_MPC(m):
    # choose the optimization solver and solve
    m.solver("ipopt")
    sol = m.solve()

    return sol
