import numpy as np
import pygame
import pyomo.environ as pyo

screen_x = 700
screen_y = 500
SCREEN_SIZE = (screen_x, screen_y)
DARK_GRAY = (50, 50, 50)
WHITE = (255, 255, 255)
YELLOW = (255, 191, 0)

pygame.init()
screen = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption("MPC")
clock = pygame.time.Clock()


def main():
    running = True

    target_time_accumulator = 0  # Accumulate time (ms)

    goal_pos = 0
    obj_pos = 0
    obj_pos_planned = []
    goal_list = [
        # (1000, 0.1),
        (5000, 0.4),
        (10000, 1.0),
        (15000, 0.1),
        (18000, 0.8),
    ]  # (timing, position)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(DARK_GRAY)

        # Render the goal
        pygame.draw.circle(screen, WHITE, (goal_pos * screen_x, int(screen_y / 2)), 10)

        if len(goal_list) > 0:
            if target_time_accumulator > goal_list[0][0]:
                goal_time, goal_pos = goal_list.pop(0)
                pos = obj_pos / screen_x
                m = mpc(pos, goal_pos)
                pyo.SolverFactory('glpk').solve(m)
                K = np.array([k for k in m.k])
                obj_pos_planned += [m.y[k]() for k in K]

        if len(obj_pos_planned) > 0:
            pos = obj_pos_planned.pop(0)
            print(f"New pos: {pos}, planned: {obj_pos_planned}")
            obj_pos = pos * screen_x

        pygame.draw.circle(screen, YELLOW, (obj_pos, 250), 10)

        target_time_accumulator += clock.get_time()
        pygame.display.flip()
        clock.tick(10)  # 60 FPS
    pygame.quit()


def mpc(pos_init=0.5, pos_target=0.7, vel_init=0, vel_target=0, N=30, h=0.1):
    """
    Pyomo example code for model predictive control based on Jeffrey C. Kantor's sample code.
    Source: https://jckantor.github.io/ND-Pyomo-Cookbook/notebooks/02.06-Model-Predictive-Control-of-a-Double-Integrator.html

    :param pos_init: value between -1 to 1
    :param pos_target:  value between -1 to 1
    :param vel_init: value between -1 to 1
    :param vel_target: value between -1 to 1
    :param N: Control time horizon
    :param h: Control time delta
    :return:
    """
    m = pyo.ConcreteModel()
    m.states = pyo.RangeSet(1, 2)
    m.k = pyo.RangeSet(0, N)

    m.h = pyo.Param(initialize=h, mutable=True)
    m.ic = pyo.Param(m.states, initialize={1: pos_init, 2: vel_init}, mutable=True)  # Initial condition
    m.gamma = pyo.Param(default=0.5, mutable=True)

    m.x = pyo.Var(m.states, m.k)
    m.icfix = pyo.Constraint(m.states, rule=lambda m, i: m.x[i, 0] == m.ic[i])  # Initialize the variable.
    m.x[1,N].fix(pos_target)  # position at time N should be 0
    m.x[2,N].fix(vel_target)  # velocity at time N should be 0

    m.u = pyo.Var(m.k, bounds=(-1, 1))
    m.upos = pyo.Var(m.k, bounds=(0, 1))
    m.uneg = pyo.Var(m.k, bounds=(0, 1))
    m.usum = pyo.Constraint(m.k, rule=lambda m, k: m.u[k] == m.upos[k] - m.uneg[k])

    m.y = pyo.Var(m.k, bounds=(-1, 1))
    m.ypos = pyo.Var(m.k, bounds=(0, 1))
    m.yneg = pyo.Var(m.k, bounds=(0, 1))
    m.ysum = pyo.Constraint(m.k, rule=lambda m, k: m.y[k] == m.ypos[k] - m.yneg[k])

    m.x1_update = pyo.Constraint(m.k, rule=lambda m, k:
                                 m.x[1, k+1] == m.x[1, k] + m.h*m.x[2, k] + (m.h**2 / 2) * m.u[k] if k < N else pyo.Constraint.Skip)
    m.x2_update = pyo.Constraint(m.k, rule=lambda m, k:
                                 m.x[2, k+1] == m.x[2, k] + m.h*m.u[k] if k < N else pyo.Constraint.Skip)

    m.y_output = pyo.Constraint(m.k, rule=lambda m, k: m.y[k] == m.x[1, k])

    m.uobj = m.gamma * sum(m.upos[k] + m.uneg[k] for k in m.k)
    m.yobj = (1-m.gamma) * sum(m.ypos[k] + m.yneg[k] for k in m.k)
    m.obj = pyo.Objective(expr=m.uobj + m.yobj, sense=pyo.minimize)
    return m


if __name__ == "__main__":
    main()