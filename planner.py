import numpy as np
from ompl import base as ob
from ompl import geometric as og


class Planner():

    def isStateValid(self, state):
        # limits check
        for i in range(self.dof):
            if self.ll[i] < state[i] < self.ul[i]:
                continue
            return False

        # collision check
        return self.collision_free(state)

    def __init__(self, dof, lower_limits, upper_limits, collision_free=lambda x: True):
        # create an SE2 state space
        space = ob.RealVectorStateSpace(dof)

        # set lower and upper bounds
        bounds = ob.RealVectorBounds(dof)
        bounds.setLow(-np.pi)
        bounds.setHigh(np.pi)
        space.setBounds(bounds)

        # create a simple setup object
        ss = og.SimpleSetup(space)
        ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.isStateValid))
        print(ss.getSpaceInformation().settings())

        # expose relevant variables
        self.ss = ss
        self.dof = dof
        self.ll = lower_limits
        self.ul = upper_limits
        self.start = ob.State(space)
        self.goal = ob.State(space)
        self.collision_free = collision_free

    def plan(self, start, goal, time=1.0, TypePlanner=og.RRTConnect):
        for i in range(self.dof):
            self.start[i] = start[i]
            self.goal[i] = goal[i]
        self.ss.setStartAndGoalStates(self.start, self.goal)
        self.ss.setPlanner(TypePlanner(self.ss.getSpaceInformation()))

        # this will automatically choose a default planner with default parameters
        solved = self.ss.solve(time)
        if solved:
            # try to shorten the path
            self.ss.simplifySolution()
            # print the simplified path
            states = self.ss.getSolutionPath().getStates()
            return [[state[i] for i in range(self.dof)] for state in states]
        return []


if __name__ == "__main__":
    planner = Planner(6, [-1.58] * 6, [1.58] * 6)
    plan = planner.plan(
        np.random.uniform(-1.57, 1.57, 6), np.random.uniform(-1.57, 1.57, 6)
    )
    print(plan)
