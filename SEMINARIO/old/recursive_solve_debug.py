import math
import pandas as pd
import numpy as np

df = pd.read_excel('Datos problema doblaje(30 tomas, 10 actores).xlsx', header = 1, index_col = 0)

# Drop columns and rows to keep only the data
data = df[:-2].copy() # drop two last rows
data.drop(columns=["Unnamed: 11", "Total"], inplace=True)  # drop two last columns
data = data.astype(int)

# Restrictions
max_shots = 6

min_days = 1 + (max(np.sum(data, axis = 0)) // max_shots)

def is_valid(sol: np.array, data: np.array, max_shots: int = 6):
    # comprobar que se graben todas las tomas una vez
    if any(np.sum(sol, axis=0) != 1): return False

    # comprobar que en un día no se requiera a un actor más de {max_shots} veces
    if any([any((sol[day] @ data) > max_shots) for day in range(len(sol))]): return False

    return True

def unavailable_slots(solution, data, max_shots) -> np.array(bool) :
    return (solution @ data == max_shots)

def score_by_day(shot, solution, data, max_shots) -> np.array(int):
    '''
    Returns the fitting score of a shot in a given day.

    The fitting score is the measure of how well a shot would fit in a day.
    Measured with a logical XOR operation between the available actor slots
    (from func unavailable_slots()) and the required actors in "shot".
    '''
    return np.sum(np.logical_xor(shot, unavailable_slots(solution, data, max_shots)), axis=1)


def recursive_solve(solution, data, max_shots):
    is_pending = np.logical_not(np.sum(solution, axis=0))

    if any(is_pending):
        pending_shots = data[:][is_pending]  # Get only the shots that are still not in the solution

        scores = []
        for _, shot in pending_shots.iterrows():
            scores.append(score_by_day(np.array(shot), solution, data, max_shots))

        scores = np.array(scores).T

        day, i = np.unravel_index(np.argmax(scores), scores.shape)  # Get the day and shot that best fit together
        shot_id = pending_shots.index[i] - 1
        solution[day, shot_id] = 1

        recursive_solve(solution, data, max_shots)

    return solution

def solve(data, max_shots):

    min_days = 1 + (max(np.sum(data, axis = 0)) // max_shots)
    solution = np.zeros((min_days, len(data)), dtype=int)
    recursive_solve(solution, data, max_shots)
    return solution

sol = solve(data, 6)

a = 1