import os
import math
import random

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


def compute_remaining_demand(demand):
    remaining_demand = 0
    for dem in demand:
        remaining_demand = remaining_demand + dem.count
    return remaining_demand


def compute_min_cost(demand):
    cost = 0.0
    for dem in demand:
        cost = cost + dem.count * dem.distance
    return cost


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
    def __init__(self, start_place, places):
        self.places = places  # reference to a mapping of place id -> position
        self.position = places[start_place]
        self.traveled_distance = 0.0
        self.path = []  # traveled path (start,end)
        self.jobs = []  # assigned jobs (Job objects)
        self.cargo = 0
        self.node = start_place

    def add_job(self, destination, load, unload, distance_to_cover):
        self.jobs.append(Job(destination, load, unload))
        self.traveled_distance = self.traveled_distance + distance_to_cover
        self.position = self.places[destination]
        self.node = destination

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
        ret = Vehicle(self.node, self.places)
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

        # find the longest straight line between two places
        max_diameter = 0.0
        for x in self.places.values():
            for y in self.places.values():
                max_diameter = max(max_diameter, distance(x, y))
        self.map_diameter = max_diameter
        self.termination_request = False

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

    def upper_bound(self, remaining_demand, offset):
        return offset + (2 * remaining_demand) * self.map_diameter

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
        vehicles = {identifier: Vehicle(identifier, self.places) for identifier in
                    range(1, num_vehicles + 1)}
        ret = self.continue_greedy_paths(demand, vehicles)
        """fuse jobs, validation fails without this"""
        for vi in ret:
            ret[vi].fuse_jobs()  # this combines unload and load jobs on the same machine
        return ret

    """
    finds the optimal schedule by using a global optimization method, use this for one vehicle only, 
    this gets way to complex for more vehicles
    """

    def branch_and_bound(self, num_vehicles, initial_bound=float('inf'), exploration_limit=-1):
        """what is the upper bound of a given incomplete schedule?"""
        # 1. find the longest path and take it times the remaining jobs, guranteed to be equal or worse than the worst

        """are there other schedules to exclude?"""
        # always pick something up if possible, ignore schedules not doing this

        """ this represents one state of the computation and will be stored as permuted copy again in the branch steps"""
        demand = {di: Demand(d[0], d[1], d[2], self.place_distance(d[0], d[1])) for di, d in enumerate(self.demand) if
                  d[2] > 0}
        vehicles = {identifier: Vehicle(identifier, self.places) for identifier in
                    range(1, num_vehicles + 1)}

        first_bound = compute_min_cost(demand.values())
        # demand, vehicles, score, lower bound
        queue = [(demand, vehicles, 0.0, first_bound)]

        best_schedule = (None, float('inf'))  # contains a map of vehicle objects and a score for the schedule
        bound = initial_bound  # this will be update while searching schedules

        progress = 0
        completed_schedules = 0
        passed_branch = 0
        bound_changed = True

        pop_index = -1  # -1 for dfs, 0 for bfs

        bfs_to_dfs_ratio = 0.00
        min_lower_bound = compute_min_cost(demand.values()) / num_vehicles
        last_passed_lower_bound = initial_bound
        last_bound_change = 0
        allways_try_non_local_demand = False

        while len(queue) > 0 and not self.termination_request:
            progress = progress + 1
            current_demand, current_vehicles, score, _ = queue.pop(pop_index)
            """ bound , cut branches violating the boundaries """
            # check if schedule is still in bounds (may changed because of an update)
            if score > bound:
                continue

            """ branch , generate new tree nodes"""
            # find jobs suitable for the vehicles and create new states
            for vehicle_id, vehicle in current_vehicles.items():
                local_demand = [(k, d) for k, d in current_demand.items() if d.start == vehicle.node]
                accepted_demand_1 = local_demand if len(local_demand) > 0 else current_demand.items()
                accepted_demand = [x for x in accepted_demand_1]
                random.shuffle(accepted_demand)
                for demand_id, demand_value in accepted_demand:

                    if demand_value.count > 0:
                        """add that job to the vehicles tasks and deal with these data object if they where immutable"""
                        new_vehicle = vehicle.copy()
                        new_vehicle.add_job(demand_value.start, True, False,
                                            distance(vehicle.position, self.places[demand_value.start]))
                        new_vehicle.add_job(demand_value.end, False, True, demand_value.distance)

                        # create new states
                        new_demand = current_demand.copy()
                        """add updated demand to new state"""
                        new_dem_count = demand_value.count - 1
                        if new_dem_count <= 0:
                            new_demand.pop(demand_id)
                        else:
                            d = demand_value.copy()
                            d.count = new_dem_count
                            new_demand[demand_id] = d
                            assert (new_demand[demand_id].count > 0)
                        """replace old vehicle object"""
                        new_schedule = current_vehicles.copy()
                        new_schedule[vehicle_id] = new_vehicle
                        new_score = rate_schedule(new_schedule.values())  # best case == score
                        lower_bound = (new_score + compute_min_cost(new_demand.values())) / num_vehicles
                        if len(local_demand) == 0:
                            lower_bound = lower_bound + min([self.place_distance(new_vehicle.node, v.start) for k, v in accepted_demand])
                        """ bound step for generated items"""
                        remaining_demand = compute_remaining_demand(new_demand.values())

                        if lower_bound >= bound:
                            continue
                        passed_branch = passed_branch + 1
                        last_passed_lower_bound = lower_bound

                        # update boundaries
                        # new_bound = self.upper_bound(remaining_demand, new_score)
                        # new_bound = new_score + 2 * compute_min_cost(new_demand.values()) + self.cost_to_fulfill(
                        #    new_vehicle.position, demand_value)
                        # if new_bound < bound:
                        #    bound = new_bound
                        #    annealing_bound = new_bound
                        #    bound_changed = True

                        # finished?
                        if remaining_demand == 0:
                            completed_schedules = completed_schedules + 1
                            artificial_annealing = True  # activate annealing
                            if new_score < best_schedule[1]:
                                best_schedule = (new_schedule, new_score)
                                bound = new_score
                                bound_changed = True
                                last_bound_change = progress
                            continue

                        # add to queue
                        new_state = (new_demand, new_schedule, new_score, lower_bound)
                        queue.append(new_state)
                    else:
                        print("found demand with count 0")

            #if len(queue) > 100000:
            #    if passed_branch > 900 and completed_schedules > 0:
            #        passed_branch = 0
                    # if random.randint(0, 1000) > 999:
                    #    random.shuffle(queue)
                    # annealing step
                    # if artificial_annealing:
                    #    annealing_bound = max(min_lower_bound+self.map_diameter, annealing_bound * 0.99)
            # console output during computation
            if progress % 100000 == 0 or bound_changed:
                print(
                    f"mode: {'bfs' if pop_index == 0 else 'dfs'} queue size: {len(queue)} bound: {bound} complete: {completed_schedules} last accepted lower bound/min bound: {last_passed_lower_bound}/{min_lower_bound}    ",
                    end="\r")
                if bound_changed:
                    queue = [(d, c, s, lb) for (d, c, s, lb) in queue if lb < bound]
                    bound_changed = False
                c = random.randint(0, 20)
                if c == 0:
                    random.shuffle(queue)
                #elif c == 1:
                #    queue.sort(key=lambda x: x[3], reverse=True)

            if bfs_to_dfs_ratio > 0.0:
                progress_limit = max(10000, int(2000 * math.log(len(queue) + 1, 10)))
                progress_limit = int(bfs_to_dfs_ratio * progress_limit if pop_index == 0 else progress_limit)
                if progress % progress_limit == 0:
                    if completed_schedules > 0:
                        pop_index = 0 if pop_index == -1 else -1
                    else:
                        pop_index = -1
            if exploration_limit > 0 and progress-last_bound_change > exploration_limit:
                self.terminate()
                print("terminating due to exploration limit hit")

        """fuse jobs, validation fails without this"""
        if best_schedule[0] is not None:
            for vi in best_schedule[0]:
                best_schedule[0][vi].fuse_jobs()  # this combines unload and load jobs on the same machine
        """ return the results """
        return best_schedule[0]

    def terminate(self):
        self.termination_request = True


def main():
    num_vehicles = 1
    routes = TransportRoutes("transport_demand.txt", "nodes.csv")
    # print(routes.demand)
    # print(routes.places)
    # vehicles = routes.greedy_paths(num_vehicles)
    greedy_schedule = routes.greedy_paths(num_vehicles)
    vehicles = routes.branch_and_bound(num_vehicles, initial_bound=rate_schedule(greedy_schedule.values())+1.0, exploration_limit=100000000)
    #vehicles = routes.branch_and_bound(num_vehicles, 7193.8964)

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
