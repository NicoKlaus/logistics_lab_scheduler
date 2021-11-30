import os
import math


def read_node_file(f):
    lines = f.readlines()
    values = [line.replace("\n", "").split(",") for line in lines]
    return {int(t[0]): (float(t[1]), float(t[2])) for t in values}


def distance(a, b):
    return length((a[0] - b[0], a[1] - b[1]))


def length(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1])


class Demand:
    def __init__(self, start, end, count, path_distance):
        self.start = start
        self.end = end
        self.distance = path_distance  # distance to cover between the two points given by start and end
        self.count = count


class Job:
    def __init__(self, destination, op_load, op_unload):
        self.destination = destination
        self.load = op_load
        self.unload = op_unload


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
        self.traveled_distance = self.traveled_distance+distance_to_cover
        self.position = self.places[destination]

    def fuse_jobs(self):
        job_list = []
        i = 0
        while i < len(self.jobs)-1:
            if self.jobs[i].destination == self.jobs[i + 1].destination:
                if (self.jobs[i].load ^ self.jobs[i+1].load) and (self.jobs[i].unload ^ self.jobs[i+1].unload):
                    job_list.append(Job(self.jobs[i].destination, 1, 1))
                    i = i + 2
                    continue
            job_list.append(self.jobs[i])
            i = i + 1
        if i < len(self.jobs):
            job_list.append(self.jobs[i])
        self.jobs = job_list

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
    def greedy_paths(self, num_vehicles):
        """ create demand objects with precomputed path lengths"""
        demand = {di: Demand(d[0], d[1], d[2], self.place_distance(d[0], d[1])) for di, d in enumerate(self.demand)}
        vehicles = {identifier: Vehicle(self.places[identifier], self.places) for identifier in range(1, num_vehicles + 1)}

        i = 0
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
            dem.count = dem.count-1
            if dem.count <= 0:
                demand.pop(dem_i)
        """fuse jobs, validation fails without this"""
        for vi in vehicles:
            vehicles[vi].fuse_jobs()  # this combines unload and load jobs on the same machine
        return vehicles


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    routes = TransportRoutes("transport_demand.txt", "nodes.csv")
    print(routes.demand)
    print(routes.places)
    vehicles = routes.greedy_paths(1)

    fo = open("schedule.txt", "w")
    fo.write("VehicleId;Location;unload;load\n")
    for vi in vehicles:
        for job in vehicles[vi].jobs:
            line = "" + str(vi) + ";" + str(job.destination) + ";" + str(int(job.unload)) + ";" + str(int(job.load)) + "\n"
            fo.write(line)
    fo.close()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
