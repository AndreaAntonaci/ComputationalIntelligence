from random import random
from functools import reduce
from collections import namedtuple
from queue import PriorityQueue, SimpleQueue, LifoQueue
import numpy as np
import itertools
import timeit
import pandas as pd
import multiprocessing


State = namedtuple('State', ['taken', 'not_taken'])
def goal_check(state,SETS,PROBLEM_SIZE):
    return np.all(reduce(np.logical_or, [SETS[i] for i in state.taken], np.array([False for _ in range(PROBLEM_SIZE)])))


def pruning(SETS,PROBLEM_SIZE,NUM_SETS):
    def cost_check(state,SETS,PROBLEM_SIZE): #That's my cost function for least overlap
        return sum(valX == True for sX in state.taken for valX in SETS[sX])
    start_time = timeit.default_timer()
    frontier = LifoQueue()
    frontier.put(State(set(), set(range(NUM_SETS))))
    counter = 0
    best_state = None
    best_cost = 1000000000000000000000000000000000000000000000000000000000000
    #JUICY STUFF with Pruning
    while not frontier.empty():
        current_state = frontier.get()
        counter += 1

        #searching for a new lower cost
        curr_cost = cost_check(current_state,SETS,PROBLEM_SIZE)
        #if my cost is higher than the best, I won't go on on that path nor I check if it's a goal state
        if curr_cost < best_cost:
            #if I have reached the goal:
            if goal_check(current_state,SETS,PROBLEM_SIZE):
                if curr_cost < best_cost:
                    best_cost = curr_cost
                    best_state = current_state
            else:
                #adding new states to the frontier
                for action in current_state[1]:
                    new_state = State(current_state.taken ^ {action}, current_state.not_taken ^ {action})
                    frontier.put(new_state)
        else:
            pass #useless: just to make more clear that it does nothing
        #OUTCOMES
    end_time = timeit.default_timer()
    execution_time = end_time - start_time
    return counter,best_cost,execution_time


def dijkstra(SETS,PROBLEM_SIZE,NUM_SETS):
    def cost_check(state,SETS): #That's my cost function for least overlap
        return sum([ sum([ 1 for valX in SETS[sX] if valX == True ]) for sX in state.taken])
    #SET UP
    start_time = timeit.default_timer()
    frontier = PriorityQueue()
    frontier.put([0,State(set(), set(range(NUM_SETS)))])
    counter = 0
    #JUICY STUFF
    while not frontier.empty(): #I don't need Pruning anymore because the first goal state will be the best one
        current_state = frontier.get()[1]
        counter += 1
        #if I have reached the goal:
        if goal_check(current_state,SETS,PROBLEM_SIZE):
            break
        else:
            #adding new states to the frontier
            for action in current_state[1]:
                new_state = State(current_state.taken ^ {action}, current_state.not_taken ^ {action})
                new_cost = cost_check(new_state,SETS)
                frontier.put([new_cost, new_state])
    end_time = timeit.default_timer()
    execution_time = end_time - start_time
    return counter,cost_check(current_state,SETS),execution_time

def A_star(SETS,PROBLEM_SIZE,NUM_SETS):
    def cost_checkA(state,SETS): #That's my cost function for least overlap
        return sum([ sum([ 1 for valX in SETS[sX] if valX == True ]) for sX in state.taken]) + \
            sum([ 1 for sX in reduce(np.logical_or, [SETS[i] for i in state.taken], np.array([False for _ in range(PROBLEM_SIZE)]))
                if sX is False])
    #SET UP
    start_time = timeit.default_timer()
    frontier = PriorityQueue()
    frontier.put([0,State(set(), set(range(NUM_SETS)))])
    counter = 0
    #JUICY STUFF
    while not frontier.empty(): #I don't need Pruning anymore because the first goal state will be the best one
        current_state = frontier.get()[1]
        counter += 1
        #if I have reached the goal:
        if goal_check(current_state,SETS,PROBLEM_SIZE):
            break
        else:
            #adding new states to the frontier
            for action in current_state[1]:
                new_state = State(current_state.taken ^ {action}, current_state.not_taken ^ {action})
                new_cost = cost_checkA(new_state,SETS)
                frontier.put([new_cost, new_state])
    end_time = timeit.default_timer()
    execution_time = end_time - start_time
    return counter,cost_checkA(current_state,SETS),execution_time


def worker(combo, results_list, status, end, runNum,limit):
    PROBLEM_SIZE, NUM_SETS, BERN_PDF_P = combo
    while True:
        SETS = tuple(np.array([random() < BERN_PDF_P for _ in range(PROBLEM_SIZE)]) for _ in range(NUM_SETS))
        if(goal_check(State(set(range(NUM_SETS)), set()),SETS,PROBLEM_SIZE)):
            break
    res=pruning(SETS,PROBLEM_SIZE,NUM_SETS)
    results_list.append(['Pruning','N','N',runNum+1,PROBLEM_SIZE,NUM_SETS,BERN_PDF_P,res[0],res[1],res[2]])
    res=dijkstra(SETS,PROBLEM_SIZE,NUM_SETS)
    results_list.append(['Dijkstra','N','N',runNum+1,PROBLEM_SIZE,NUM_SETS,BERN_PDF_P,res[0],res[1],res[2]])
    res=A_star(SETS,PROBLEM_SIZE,NUM_SETS)
    results_list.append(['A*','N','N',runNum+1,PROBLEM_SIZE,NUM_SETS,BERN_PDF_P,res[0],res[1],res[2]])
    SETS_UNIQUE=tuple(set(map( lambda x: tuple(x) , SETS))) #remove duplicates
    res=pruning(SETS_UNIQUE,PROBLEM_SIZE,len(SETS_UNIQUE))
    results_list.append(['Pruning','Y','N',runNum+1,PROBLEM_SIZE,NUM_SETS,BERN_PDF_P,res[0],res[1],res[2]])
    res=dijkstra(SETS_UNIQUE,PROBLEM_SIZE,len(SETS_UNIQUE))
    results_list.append(['Dijkstra','Y','N',runNum+1,PROBLEM_SIZE,NUM_SETS,BERN_PDF_P,res[0],res[1],res[2]])
    res=A_star(SETS_UNIQUE,PROBLEM_SIZE,len(SETS_UNIQUE))
    results_list.append(['A*','Y','N',runNum+1,PROBLEM_SIZE,NUM_SETS,BERN_PDF_P,res[0],res[1],res[2]])
    SETS_UNIQUE_SORTED=sorted(SETS_UNIQUE,key=lambda x:np.add.reduce(x),reverse=True) #sort sets
    res=pruning(SETS_UNIQUE_SORTED,PROBLEM_SIZE,len(SETS_UNIQUE))
    results_list.append(['Pruning','Y','Y',runNum+1,PROBLEM_SIZE,NUM_SETS,BERN_PDF_P,res[0],res[1],res[2]])
    res=dijkstra(SETS_UNIQUE_SORTED,PROBLEM_SIZE,len(SETS_UNIQUE))
    results_list.append(['Dijkstra','Y','Y',runNum+1,PROBLEM_SIZE,NUM_SETS,BERN_PDF_P,res[0],res[1],res[2]])
    res=A_star(SETS_UNIQUE_SORTED,PROBLEM_SIZE,len(SETS_UNIQUE))
    results_list.append(['A*','Y','Y',runNum+1,PROBLEM_SIZE,NUM_SETS,BERN_PDF_P,res[0],res[1],res[2]])
    status.value=status.value+1
    print(f"Run n. {status.value} su {end}")
    limit.release()




if __name__ == '__main__':
    cols = ['Algoritmo', 'Unique(Y/N)', 'Sorted(Y/N)', 'Run#', 'Problem_Size', '#_Sets', 'Probability', '#_of_Steps', 'Best_Cost', 'Time']
    numberofRuns = 50  # Modify as needed
    PROBLEM_SIZES = [5]
    NUMS_SETS = [30]
    BERN_PDF_PS = [0.4]
    combinations = list(itertools.product(PROBLEM_SIZES, NUMS_SETS, BERN_PDF_PS))
    State = namedtuple('State', ['taken', 'not_taken'])
    results_list = multiprocessing.Manager().list()
    status = multiprocessing.Value('i', 0)
    end = len(combinations) * numberofRuns
    max_processes = 8  # Limit the number of concurrent processes
    limit = multiprocessing.Semaphore(max_processes)
    processes = []

    for combo in combinations:
        for runNum in range(numberofRuns):
            limit.acquire()
            p = multiprocessing.Process(target=worker, args=(combo, results_list, status, end, runNum,limit))
            processes.append(p)
            p.start()

    for p in processes:
        p.join()
    results = pd.DataFrame(list(results_list), columns=cols)
    results.to_csv('results.csv', index=False)