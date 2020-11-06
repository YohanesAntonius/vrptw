from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import math
import numpy as np
import matplotlib.pyplot as plt


def create_data_model():
    """Stores the data for the problem."""
    data = {}
    data['time_matrix'] = [
        [0, 5, 9, 3, 10, 6, 8, 7, 5, 1, 10, 6, 2, 6, 2, 5, 5, 6, 8, 3, 7, 6, 7, 9, 8, 8],
        [5, 0, 2, 2, 4, 8, 7, 8, 5, 9, 2, 4, 1, 7, 3, 4, 6, 2, 6, 9, 3, 8, 6, 9, 6, 3],
        [9, 2, 0, 2, 6, 7, 9, 8, 9, 1, 4, 6, 5, 3, 1, 9, 4, 4, 2, 10, 9, 3, 4, 4, 5, 10],
        [3, 2, 2, 0, 5, 1, 6, 7, 2, 5, 4, 8, 10, 1, 9, 8, 2, 10, 2, 9, 4, 8, 8, 6, 8 ,9],
        [10, 4, 6, 5, 0, 7, 6, 1, 1, 6, 7, 7, 5, 1, 3, 9, 5, 6, 4, 10, 2, 7, 5, 3, 6, 3],
        [6, 8, 7 ,1, 7, 0, 2, 1, 9, 2, 5, 5, 10, 10, 4, 5, 5, 7, 2, 5, 1, 7, 3, 8, 1, 7],
        [8, 7, 9, 6, 6, 2, 0, 9, 10, 9, 1, 4, 9, 9, 10, 8, 4, 2, 8, 2, 4, 5, 5, 3, 2, 10],
        [7, 8, 8, 7, 1, 1, 9, 0, 5, 2, 3, 9, 1, 1, 6, 6, 6, 1, 4, 8, 7, 5, 7, 5, 2, 5],
        [5, 5, 9, 2, 1, 9, 10, 5, 0, 9, 3, 9, 9, 9, 3, 9, 5, 8, 10, 5, 4, 8,8, 10, 10, 2],
        [1, 9, 1, 5, 6, 2, 9, 2, 9, 0, 2, 1, 9 ,7, 8, 7, 8, 7, 7, 5, 5, 9, 5, 5, 8, 3],
        [10, 2, 4, 4, 7, 5, 1, 3, 3, 2, 0, 9, 1, 9, 1, 1, 10, 2, 4, 2, 8,6, 7, 9, 3, 6],
        [6, 4, 6, 8, 7 ,5, 4, 9, 9, 1 ,9, 0, 2, 5, 7, 4, 10, 9, 4, 2, 10, 7, 9, 2, 5, 4],
        [2, 1, 5, 10, 5, 10, 9, 1, 9, 9 ,1, 2, 0, 7, 7, 8, 5, 4, 9, 4, 9, 5, 1, 1, 8, 2],
        [6, 7, 3, 1, 1, 10, 9, 1, 9, 7, 9, 5, 7, 0, 3, 6, 7, 6, 6, 5, 5, 2, 8, 6, 3, 9],
        [2, 3, 1, 9, 3, 4, 10, 6, 3, 8, 1, 7, 7, 3, 0, 6, 1, 8, 9, 4, 3, 5, 10, 1, 8, 1],
        [5, 4, 9, 8, 9, 5, 8, 6, 9, 7, 1 ,4, 8, 6, 6, 0, 8, 8, 2, 3, 1, 3, 5, 8, 2, 8],
        [5, 6, 4, 2, 5, 5, 4, 6, 5, 8, 10, 10, 5, 7, 1, 8, 0, 6 ,10, 10, 7, 6, 3, 9, 3, 1],
        [6, 2, 4, 10, 6, 7, 2, 1, 8, 7, 2, 9, 4, 6, 8, 8, 6, 0, 2, 5, 7, 9, 7, 8, 9, 1],
        [8, 6, 2, 2, 4, 2, 8, 4, 10, 7, 4, 4, 9, 6, 9, 2, 10, 2, 0, 9, 7, 8, 7, 2, 8, 6],
        [3, 9, 10, 9, 10, 5, 2, 8, 5, 5, 2, 2, 4, 5, 4, 3, 10, 5, 9, 0, 3, 5, 5, 8, 7, 4],
        [7, 3, 9, 4, 2, 1, 4, 7, 4, 5, 8, 10, 9, 5, 3, 1, 7, 7, 7, 3 ,0, 7, 8, 9, 1, 1],
        [6, 8, 3, 8, 7, 7, 5, 5, 8, 9, 6, 7, 5, 2, 5, 3, 6, 9, 8, 5,7, 0, 1, 1, 9, 9],
        [7, 6, 4, 8, 5, 3, 5, 7, 8, 5, 7, 9, 1, 8, 10, 5, 3, 7, 7, 5, 8, 1, 0, 8, 7, 9],
        [9, 9, 4, 6, 3, 8, 3, 5, 10, 5, 9, 2, 1, 6, 1, 8, 9, 8, 2, 8, 9, 1, 8, 0, 7, 2],
        [8, 6, 5, 8, 6, 1, 2, 2, 10, 8, 3, 5, 8, 3, 8, 2, 3, 9, 8, 7, 1, 9, 7, 7, 0, 3],
        [8, 3, 10, 9, 3, 7, 10, 5, 2, 3, 6, 4, 2, 9, 1, 8, 1, 1, 6, 4, 1, 9, 9, 2, 3, 0],
    ]
    data['time_windows'] = [
        (5, 12),  # depot
        (8, 19),  # 1
        (3, 12),  # 2
        (0, 12),  # 3
        (11, 19),  # 4
        (16, 18),  # 5
        (8, 20),  # 6
        (5, 9),  # 7
        (6, 16),  # 8
        (2, 4),  # 9
        (17, 20),  # 10
        (2, 17),  # 11
        (4, 17),  # 12
        (5, 19),  # 13
        (15, 17),  # 14
        (14, 18),  # 15
        (1, 8),  # 16
        (5, 12), # 17
        (2, 13), # 18
        (0, 2), # 19
        (9, 12), # 20
        (8, 18), # 21
        (12, 19), # 22
        (6, 8), # 23
        (17, 20), # 24
        (10, 12), # 25
    ]
    data['num_vehicles'] = 4
    data['depot'] = 0
    return data

def compute_euclidean_distance_matrix(time_matrix):
    """Creates callback to return distance between points."""
    distances = {}
    for from_counter, from_node in enumerate(time_matrix):
        distances[from_counter] = {}
        for to_counter, to_node in enumerate(time_matrix):
            if from_counter == to_counter:
                distances[from_counter][to_counter] = 0
            else:
                # Manhattan distance
                distances[from_counter][to_counter] = (int(
                    math.hypot((from_node[0] - to_node[0]),
                               (from_node[1] - to_node[1]))))
    return distances

def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    time_dimension = routing.GetDimensionOrDie('Time')
    total_time = 0
    solutions = {}
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        vehicle_solution = []
        while not routing.IsEnd(index):
            vehicle_solution.append(manager.IndexToNode(index))
            time_var = time_dimension.CumulVar(index)
            plan_output += '{0} Time({1},{2}) -> '.format(
                manager.IndexToNode(index), solution.Min(time_var),
                solution.Max(time_var))
            index = solution.Value(routing.NextVar(index))
        time_var = time_dimension.CumulVar(index)
        plan_output += '{0} Time({1},{2})\n'.format(manager.IndexToNode(index),
                                                    solution.Min(time_var),
                                                    solution.Max(time_var))
        plan_output += 'Time of the route: {}min\n'.format(
            solution.Min(time_var))
        print(plan_output)
        total_time += solution.Min(time_var)
        solutions[vehicle_id] = vehicle_solution
    print('Total time of all routes: {}min'.format(total_time))
    return solutions

"""Solve the CVRP problem."""
# Instantiate the data problem.
data = create_data_model()

# Create the routing index manager.
manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']),
                                       data['num_vehicles'], data['depot'])

# Create Routing Model.
routing = pywrapcp.RoutingModel(manager)

time_matrix = compute_euclidean_distance_matrix(data['time_matrix'])


 # Create and register a transit callback.
def time_callback(from_index, to_index):
        """Returns the travel time between the two nodes."""
        # Convert from routing variable Index to time matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['time_matrix'][from_node][to_node]

transit_callback_index = routing.RegisterTransitCallback(time_callback)

# Define cost of each arc.
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

time = 'Time'
routing.AddDimension(
        transit_callback_index,
        0,  # allow waiting time
        20,  # maximum time per vehicle
        False,  # Don't force start cumul to zero.
        time)
time_dimension = routing.GetDimensionOrDie(time)
    # Add time window constraints for each location except depot.
for location_idx, time_window in enumerate(data['time_windows']):
        if location_idx == 0:
            continue
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(data['time_windows'][0][0], data['time_windows'][1][1])
    # Add time window constraints for each vehicle start node.
for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        time_dimension.CumulVar(index).SetRange(data['time_windows'][0][0],
                                                data['time_windows'][0][1])

    # Instantiate route start and end times to produce feasible times.
for i in range(data['num_vehicles']):
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.Start(i)))
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.End(i)))

    # Setting first solution heuristic.
search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
solution = routing.SolveWithParameters(search_parameters)

# Print solution on console.
if solution:
    solution_dict = print_solution(data, manager, routing, solution)

coordinates = [
        (12,5),  # depot
        (8, 19),  # 1
        (12,3),  # 2
        (0, 12),  # 3
        (11, 19),  # 4
        (18,16),  # 5
        (8, 20),  # 6
        (9,5),  # 7
        (6, 16),  # 8
        (4,2),  # 9
        (20,17),  # 10
        (2, 17),  # 11
        (17,4),  # 12
        (19,5),  # 13
        (15, 17),  # 14
        (18,14),  # 15
        (1, 8),  # 16
        (12,5), #17
        (2, 13), #18
        (0, 2), #19
        (12,9), #20
        (18,8), #21
        (12, 19), #22
        (6, 8), #23
        (17, 20), #24
        (10, 12) #25
]
X = np.array([x[0] for x in coordinates])
Y = np.array([x[1] for x in coordinates])

f, ax = plt.subplots(figsize = [8,6])

ax.plot(X, Y, 'ko', markersize=8)
ax.plot(X[0], Y[0], 'gX', markersize=30)

for i, txt in enumerate(coordinates):
    ax.text(X[i], Y[i], f"{i}")

vehicle_colors = ["g","r", "m", "c","y","b"]
for vehicle in solution_dict:
    ax.plot(X[solution_dict[vehicle] + [0]], Y[solution_dict[vehicle] + [0]], f'{vehicle_colors[vehicle]}--')
    
ax.set_title("Tugas 6 VRPTW Route")
plt.axis([-5, 25, -5, 25])
plt.show()
