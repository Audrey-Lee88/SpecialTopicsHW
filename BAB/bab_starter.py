import picos as pic
from picos import RealVariable
from copy import deepcopy
from heapq import *
import heapq as hq
import numpy as np
import itertools
import math
counter = itertools.count()

class BBTreeNode():
    def __init__(self, vars = [], constraints = [], objective='', prob=None):
        self.vars = vars
        self.constraints = constraints
        self.objective = objective
        self.prob = prob

    def __deepcopy__(self, memo):
        '''
        Deepcopies the picos problem
        This overrides the system's deepcopy method bc it doesn't work on classes by itself
        '''
        newprob = pic.Problem.clone(self.prob)
        return BBTreeNode(self.vars, newprob.constraints, self.objective, newprob)

    def buildProblem(self):
        '''
        Bulids the initial Picos problem
        '''
        prob=pic.Problem()

        prob.add_list_of_constraints(self.constraints)

        prob.set_objective('max', self.objective)
        self.prob = prob
        return self.prob

    def is_integral(self):
        '''
        Checks if all variables (excluding the one we're maxing) are integers
        '''
        for v in self.vars[:-1]:
            if v.value == None or abs(round(v.value) - float(v.value)) > 1e-4 :
                return False
        return True

    def branch_floor(self, branch_var):
        '''
        Makes a child where xi <= floor(xi)
        '''
        n1 = deepcopy(self)
        n1.prob.add_constraint( branch_var <= math.floor(branch_var.value) ) # add in the new binary constraint

        return n1

    def branch_ceil(self, branch_var):
        '''
        Makes a child where xi >= ceiling(xi)
        '''
        n2 = deepcopy(self)
        n2.prob.add_constraint( branch_var >= math.ceil(branch_var.value) ) # add in the new binary constraint
        return n2

    def bbsolve(self):
        '''
        Use the branch and bound method to solve an integer program
        This function should return:
            return bestres, bestnode_vars

        where bestres = value of the maximized objective function
              bestnode_vars = the list of variables that create bestres
        '''

        # these lines build up the initial problem and adds it to a heap
        root = self
        res = root.buildProblem().solve(solver='cvxopt')
        heap = [(res, next(counter), root)]
        bestres = -1e20 # a small arbitrary initial best objective value
        bestnode_vars = root.vars # initialize bestnode_vars to the root vars

        # print(heap)
        while len(heap) > 0:
            print("Heap Size: ", len(heap))
            _, _, node = hq.heappop(heap)
            try:
                node.prob.solve(solver='cvxopt')
            except:
                #if problem is infeasible, don't do that branch
                print("Infeasible")
                pass

            print("node_vars: ",  [float(v.value) for v in node.vars])
            #if the relaxed problem is bad
            if float(node.vars[-1]) < float(bestres - 1e-2):
                print("Get rid of branch")
                continue
            #if a valid solution then this is the new best
            elif node.is_integral():
                print("New Best Integral solution.")
                bestres = float(node.vars[-1])
                bestnode_vars = [float(v.value) for v in node.vars]
            #otherwise, create new branches and branch into it
            else:
                for v in node.vars:
                    # grab the first var that isn't an integer
                    if abs(round(v.value) - float(v.value)) > 1e-2:
                        # create the new branches
                        new_left_node = node.branch_floor(v)
                        new_right_node = node.branch_ceil(v)

                        # add the branches to the heap
                        hq.heappush(heap, (float(node.vars[-1]), next(counter), new_left_node))
                        hq.heappush(heap, (float(node.vars[-1]), next(counter), new_right_node))
                        break
        return bestres, bestnode_vars
