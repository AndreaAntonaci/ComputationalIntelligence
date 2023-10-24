from random import random
from functools import reduce
import numpy as np
from queue import PriorityQueue
from collections import namedtuple

# Define State as a namedtuple
State = namedtuple('State', ['taken', 'not_taken'])

# Define the goal_check function
def goal_check(state, sets, problem_size):
    return np.all(reduce(np.logical_or, [SETS[i] for i in state.taken], np.array([False for _ in range(PROBLEM_SIZE)])))

# Define the cost_check function
def cost_check(state):
    return len(state.taken)

# Define the heuristic function
def heuristic(state, sets, problem_size):
    uncovered_elements = np.logical_not(np.logical_or.reduce([sets[i] for i in state.taken], axis=0))
    return np.count_nonzero(uncovered_elements)

PROBLEM_SIZE = 8
NUM_SETS = 5
BERN_PDF_P = 0.1

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


# # Remove duplicate sets
# SETS_UNIQUE = tuple(set(map(lambda x: tuple(x), SETS)))

# # Sort sets by size
# SETS_UNIQUE_SORTED = sorted(SETS_UNIQUE, key=lambda x: np.sum(x), reverse=True)

# A* search without pruning
while not frontier.empty():
    _, current_state = frontier.get()
    counter += 1

    # Check if the goal state is reached
    if goal_check(current_state,SETS,PROBLEM_SIZE):
        break
    else:
        # Adding new states to the frontier
        for action in current_state.not_taken:
            new_state = State(current_state.taken ^ {action}, current_state.not_taken ^ {action})
            new_cost = cost_check(new_state)
            heuristic_value = heuristic(new_state, SETS, PROBLEM_SIZE)
            total_cost = new_cost + heuristic_value
            frontier.put([total_cost, new_state])

#OUTCOMES
print(f"Solved in {counter:,} steps")
print(cost_check(current_state))
print(current_state)


# Remove duplicate sets
SETS_UNIQUE = tuple(set(map(lambda x: tuple(x), SETS)))

# Sort sets by size
#SETS_UNIQUE_SORTED = sorted(SETS_UNIQUE, key=lambda x: np.sum(x), reverse=True)
SETS_UNIQUE_SORTED=SETS_UNIQUE
# Initialize the priority queue
frontier = PriorityQueue()
initial_state = State(set(), set(range(len(SETS_UNIQUE_SORTED))))
heuristic_value = heuristic(initial_state, SETS_UNIQUE_SORTED, PROBLEM_SIZE)
frontier.put([heuristic_value, initial_state])
counter = 0




# A* search without pruning
while not frontier.empty():
    _, current_state = frontier.get()
    counter += 1

    # Check if the goal state is reached
    if goal_check(current_state,SETS_UNIQUE_SORTED,PROBLEM_SIZE):
        break
    else:
        # Adding new states to the frontier
        for action in current_state.not_taken:
            new_cost = cost_check(current_state)
            new_state = State(current_state.taken ^ {action}, current_state.not_taken ^ {action})
            heuristic_value = heuristic(new_state, SETS_UNIQUE_SORTED, PROBLEM_SIZE)
            total_cost = new_cost + heuristic_value
            frontier.put([total_cost, new_state])

#OUTCOMES
print(f"Solved in {counter:,} steps")
print(cost_check(current_state))
print(current_state)
for set in SETS:
    print(set)