from random import random
from functools import reduce
import numpy as np
from queue import PriorityQueue
from collections import namedtuple
from math import ceil

# Define State as a namedtuple
State = namedtuple('State', ['taken', 'not_taken'])

# Define the goal_check function
def goal_check(state, sets, problem_size):
    return np.all(covered(state,sets,problem_size))

# Define the cost_check function
def cost_check(state):
    return len(state.taken)

def covered(state,sets,problem_size):
    return reduce(
        np.logical_or,
        [sets[i] for i in state.taken],
        np.array([False for _ in range(problem_size)]),
    )


def heuristic(state, sets, problem_size):#third heuristic dissussed by the professor
    already_covered = covered(state,sets,problem_size)
    if np.all(already_covered):
        return 0
    missing_size = problem_size - sum(already_covered)
    candidates = sorted((sum(np.logical_and(s, np.logical_not(already_covered))) for s in sets), reverse=True)
    taken = 1
    while sum(candidates[:taken]) < missing_size:
        taken += 1
    return taken


PROBLEM_SIZE = 50
NUM_SETS = 50
BERN_PDF_P = 0.4

# Generate random sets until a goal state is found
while True:
    SETS = tuple(np.array([random() < BERN_PDF_P for _ in range(PROBLEM_SIZE)]) for _ in range(NUM_SETS))
    if goal_check(State(set(range(NUM_SETS)), set()), SETS, PROBLEM_SIZE):
        break

# Initialize the priority queue
frontier = PriorityQueue()
initial_state = State(set(), set(range(NUM_SETS)))
heuristic_value = heuristic(initial_state, SETS, PROBLEM_SIZE)
frontier.put([heuristic_value, initial_state])
counter = 0


# A* search without pruning
while not frontier.empty():
    _, current_state = frontier.get()
    counter += 1

    # Check if the goal state is reached
    if goal_check(current_state,SETS,PROBLEM_SIZE):
        break
    else:
        # Adding new states to the frontier
        for action in current_state[1]:
            new_state = State(current_state.taken ^ {action}, current_state.not_taken ^ {action})
            new_cost = cost_check(new_state)
            heuristic_value = heuristic(new_state, SETS, PROBLEM_SIZE)
            total_cost = new_cost + heuristic_value
            frontier.put([total_cost, new_state])

#OUTCOMES
print(f"Solved in {counter:,} steps")
print(cost_check(current_state))
print(current_state)
#print(SETS)


###Now I'll try to remove any duplicate sets hoping to reduce the computational time

# Remove duplicate sets
SETS_UNIQUE = tuple(set(map(lambda x: tuple(x), SETS)))


# Initialize the priority queue
# Initialize the priority queue
frontier = PriorityQueue()
initial_state = State(set(), set(range(len(SETS_UNIQUE))))
heuristic_value = heuristic(initial_state, SETS_UNIQUE, PROBLEM_SIZE)
frontier.put([heuristic_value, initial_state])
counter = 0



# A* search without pruning
while not frontier.empty():
    current_state = frontier.get()[1]
    counter += 1

    # Check if the goal state is reached
    if goal_check(current_state,SETS_UNIQUE,PROBLEM_SIZE):
        break
    else:
        # Adding new states to the frontier
        for action in current_state[1]:
            new_state = State(current_state.taken ^ {action}, current_state.not_taken ^ {action})
            new_cost = cost_check(new_state)
            heuristic_value = heuristic(new_state, SETS_UNIQUE, PROBLEM_SIZE)
            total_cost = new_cost + heuristic_value
            frontier.put([total_cost, new_state])

#OUTCOMES
print(f"Solved in {counter:,} steps")
print(cost_check(current_state))
print(current_state)  #we see different solution because when a set is used to remove duplicates the order changes 
#print(SETS_UNIQUE)


###Now I'll try to order the set by initial coverage
###Another interesting approach could be to take first a set that covers a lot and then try to take little sets to fill the gaps

# Sort sets by size
SETS_UNIQUE_SORTED = sorted(SETS_UNIQUE, key=lambda x: np.sum(x), reverse=True)


# Initialize the priority queue
frontier = PriorityQueue()
initial_state = State(set(), set(range(len(SETS_UNIQUE_SORTED))))
heuristic_value = heuristic(initial_state, SETS_UNIQUE_SORTED, PROBLEM_SIZE)
frontier.put([heuristic_value, initial_state])
counter = 0



# A* search without pruning
while not frontier.empty():
    current_state = frontier.get()[1]
    counter += 1

    # Check if the goal state is reached
    if goal_check(current_state,SETS_UNIQUE_SORTED,PROBLEM_SIZE):
        break
    else:
        # Adding new states to the frontier
        for action in current_state[1]:
            new_state = State(current_state.taken ^ {action}, current_state.not_taken ^ {action})
            new_cost = cost_check(new_state)
            heuristic_value = heuristic(new_state, SETS_UNIQUE_SORTED, PROBLEM_SIZE)
            total_cost = new_cost + heuristic_value
            frontier.put([total_cost, new_state])

#OUTCOMES
print(f"Solved in {counter:,} steps")
print(cost_check(current_state))
print(current_state)  #we see different solution because when sorting obviusly the order changes
