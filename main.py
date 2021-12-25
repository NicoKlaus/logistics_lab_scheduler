import os
import math

import functools as ft


def read_node_file(f):
    lines = f.readlines()
    values = [line.replace("\n", "").split(",") for line in lines]
    return {int(t[0]): (float(t[1]), float(t[2])) for t in values}


def distance(a, b):
    return length((a[0] - b[0], a[1] - b[1]))


def length(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1])


# expects an iterable of vehicles as parameter
def rate_schedule(vehicles):
    ret = 0.0
    for vehicle in vehicles:
        ret = max(vehicle.traveled_distance, ret)
    return ret


class Demand:
    def __init__(self, start, end, count, path_distance):
        self.start = start
        self.end = end
        self.distance = path_distance  # distance to cover between the two points given by start and end
        self.count = count

    def copy(self):
        ret = Demand(self.start, self.end, self.count, self.distance)
        return ret


class Job:
    def __init__(self, destination, op_load, op_unload):
        self.destination = destination
        self.load = op_load
        self.unload = op_unload

    def copy(self):
        ret = Job(self.destination, self.load, self.unload)
        return ret


class Vehicle:
    def __init__(self, position, places):
        self.places = places  # reference to a mapping of place id -> position
        self.position = position
        self.traveled_distance = 0.0
        self.path = []  # traveled path (start,end)
        self.jobs = []  # assigned jobs (Job objects)
        self.cargo = 0

    def add_job(self, destination, load, unload, distance_to_cover):
        self.jobs.append(Job(destination, load, unload))
        self.traveled_distance = self.traveled_distance + distance_to_cover
        self.position = self.places[destination]

    def fuse_jobs(self):
        job_list = []
        i = 0
        while i < len(self.jobs) - 1:
            if self.jobs[i].destination == self.jobs[i + 1].destination:
                if (self.jobs[i].load ^ self.jobs[i + 1].load) and (self.jobs[i].unload ^ self.jobs[i + 1].unload):
                    job_list.append(Job(self.jobs[i].destination, 1, 1))
                    i = i + 2
                    continue
            job_list.append(self.jobs[i])
            i = i + 1
        if i < len(self.jobs):
            job_list.append(self.jobs[i])
        self.jobs = job_list

    def copy(self):
        ret = Vehicle(self.position, self.places)
        ret.jobs = self.jobs.copy()
        ret.path = self.path.copy()
        ret.traveled_distance = self.traveled_distance
        ret.cargo = self.cargo
        return ret


class TransportRoutes:
    def __init__(self, demand_file, node_file):
        f = open(demand_file, "r")
        """load demand"""
        self.demand = self.read_demand_file(f)
        f.close()

        """load places"""
        fn = open(node_file, "r")
        self.places = read_node_file(fn)
        fn.close()

    def read_demand_file(self, f):
        lines = f.readlines()
        tokens = lines[0].replace("\n", "").split(";")
        self.attribute_map = {token: i for i, token in enumerate(tokens)}
        values = [self.parse_line_td(line.replace("\n", "")) for line in lines[1:]]
        return values

    def parse_line_td(self, line):
        tokens = line.split(";")
        t = [int(t) for t in tokens]
        return t[self.attribute_map["start"]], t[self.attribute_map["dest"]], t[self.attribute_map["number"]]

    """compute distance between two places"""

    def place_distance(self, p1, p2):
        return length((self.places[p1][0] - self.places[p2][0], self.places[p1][1] - self.places[p2][1]))

    def cost_to_fulfill(self, start_position, job: Demand):
        return distance(start_position, self.places[job.start]) + job.distance

    """
    finds a schedule for each vehicle using a greedy optimization tactic
    """

    def continue_greedy_paths(self, demand, vehicles):
        while len(demand) > 0:
            """find vehicle with the least distance traveled by now"""
            vi = min(vehicles, key=lambda x: vehicles[x].traveled_distance)
            vehicle = vehicles[vi]
            """find the job which requires the least distance to cover for that vehicle"""
            dem_i = min(demand, key=lambda k: self.cost_to_fulfill(vehicle.position, demand[k]))
            dem = demand[dem_i]
            if dem.count > 0:
                """add that job to the vehicles tasks"""
                vehicle.add_job(dem.start, True, False, distance(vehicle.position, self.places[dem.start]))
                vehicle.add_job(dem.end, False, True, dem.distance)
            """remove demand"""
            dem.count = dem.count - 1
            if dem.count <= 0:
                demand.pop(dem_i)
        """fuse jobs, validation fails without this"""
        # for vi in vehicles:
        #     vehicles[vi].fuse_jobs()  # this combines unload and load jobs on the same machine
        return vehicles

    def greedy_paths(self, num_vehicles):
        """ create demand objects with precomputed path lengths"""
        demand = {di: Demand(d[0], d[1], d[2], self.place_distance(d[0], d[1])) for di, d in enumerate(self.demand)}
        vehicles = {identifier: Vehicle(self.places[identifier], self.places) for identifier in
                    range(1, num_vehicles + 1)}
        ret = self.continue_greedy_paths(demand, vehicles)
        """fuse jobs, validation fails without this"""
        for vi in ret:
            ret[vi].fuse_jobs()  # this combines unload and load jobs on the same machine
        return ret

    """
    finds the optimal schedule by using a global optimization method
    """

    def branch_and_bound(self, num_vehicles, initial_bound=float('inf')):
        """what is the upper bound of a given incomplete schedule?"""
        # 1. find the longest path and take it times the remaining jobs, guranteed to be equal or worse than the worst

        """are there other schedules to exclude?"""
        # always pick something up if possible, ignore schedules not doing this

        # find the longest straight line between two places
        max_diameter = 0.0
        for x in self.places.values():
            for y in self.places.values():
                max_diameter = max(max_diameter, distance(x, y))

        """ this represents one state of the computation and will be stored as permuted copy again in the branch steps"""
        demand = {di: Demand(d[0], d[1], d[2], self.place_distance(d[0], d[1])) for di, d in enumerate(self.demand)}
        vehicles = {identifier: Vehicle(self.places[identifier], self.places) for identifier in
                    range(1, num_vehicles + 1)}

        queue = [(demand, vehicles)]  # it is important to handle vehicles as if they were immutable objects, TODO may add score here

        best_schedule = (None, float('inf'))  # contains a map of vehicle objects and a score for the schedule
        bound = initial_bound  # this will be update while searching schedules

        progress = 0
        passed_branch = 0
        completed_schedules = 0

        while len(queue) > 0:
            state = queue.pop(0)
            """ bound , cut branches violating the boundaries """
            # TODO move bound step in loop of branch step to avoid creating many useless nodes in BFS mode
            # rate schedule
            remaining_demand = 0
            for dem in state[0].values():
                remaining_demand = remaining_demand + dem.count
            score = rate_schedule(state[1].values())  # best case == score

            # check if schedule is still in bounds
            if score > bound:
                continue

            # update boundaries
            bound = min(bound, score + (2 * remaining_demand) * max_diameter)  # worst case
            passed_branch = passed_branch + 1

            # finished?
            if remaining_demand == 0:
                completed_schedules = completed_schedules + 1
                if score < best_schedule[1]:
                    best_schedule = (state[1], score)
                    continue

            """ branch , generate new tree nodes"""
            # find jobs suitable for the vehicles and create new states
            for demand_id, demand_value in state[0].items():

                for vehicle_id, vehicle in state[1].items():

                    if demand_value.count > 0:
                        """add that job to the vehicles tasks and deal with these data object if they where immutable"""
                        new_vehicle = vehicle.copy()
                        new_vehicle.add_job(demand_value.start, True, False,
                                            distance(vehicle.position, self.places[demand_value.start]))
                        new_vehicle.add_job(demand_value.end, False, True, demand_value.distance)

                        # create new states
                        new_demand = state[0].copy()
                        """add updated demand to new state"""
                        new_dem_count = demand_value.count - 1
                        if new_dem_count <= 0:
                            new_demand.pop(demand_id)
                        else:
                            d = demand_value.copy()
                            d.count = new_dem_count
                            new_demand[demand_id] = d
                        """replace old vehicle object"""
                        new_schedule = state[1].copy()
                        new_schedule[vehicle_id] = new_vehicle
                        new_state = (new_demand, new_schedule)

                        # add to queue
                        queue.append(new_state)

                # console output during computation
                progress = progress+1
                if progress > 100000:
                    print(f"queue size: {len(queue)} bound: {bound} passed branch: {passed_branch/1000.0}% complete: {completed_schedules}    ", end="\r")
                    progress = 0
                    passed_branch = 0
        """fuse jobs, validation fails without this"""
        for vi in best_schedule[0]:
            best_schedule[0][vi].fuse_jobs()  # this combines unload and load jobs on the same machine
        """ return the results """
        return best_schedule[0]


def main():
    num_vehicles = 1
    routes = TransportRoutes("transport_demand.txt", "nodes.csv")
    # print(routes.demand)
    # print(routes.places)
    vehicles = routes.greedy_paths(num_vehicles)
    # greedy_schedule = routes.greedy_paths(num_vehicles)
    # vehicles = routes.branch_and_bound(num_vehicles, rate_schedule(greedy_schedule.values()))
    # vehicles = routes.branch_and_bound(num_vehicles)

    fo = open("schedule.txt", "w")
    fo.write("VehicleId;Location;unload;load\n")
    for vi in vehicles:
        for job in vehicles[vi].jobs:
            line = "" + str(vi) + ";" + str(job.destination) + ";" + str(int(job.unload)) + ";" + str(
                int(job.load)) + "\n"
            fo.write(line)
    fo.close()


if __name__ == '__main__':
    main()
