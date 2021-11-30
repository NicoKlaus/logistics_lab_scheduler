__author__ = "Karl-Benedikt Reith"


def validation(demand_file_name='transport_demand.txt', schedule_file_name='schedule.txt',
               machine_positions=None, velocity=1):

    if machine_positions is None:
        machine_positions = [(5, 5), (50, 5), (70, 25), (60, 40), (30, 40), (5, 25)]

    # ############## init ##########################
    demand = dict()

    # open demand file
    try:
        with open(demand_file_name) as demand_file:
            lines = demand_file.readlines()
    except FileNotFoundError:
        print('Error: can not open demand file: {}'.format(demand_file_name))
        exit()

    # remove header if exists
    try:
        int(lines[0].split(';')[0])
    except ValueError:
        lines = lines[1:]

    # create demand dictionary
    for line in lines:
        pos = list(map(int, line.rstrip('\n').split(';')))
        try:
            demand[pos[0]]
        except KeyError:
            demand[pos[0]] = dict()

        try:
            demand[pos[0]][pos[1]]
        except KeyError:
            demand[pos[0]][pos[1]] = 0

        demand[pos[0]][pos[1]] += pos[2]

    # open schedule file
    try:
        with open(schedule_file_name) as schedule_file:
            lines = schedule_file.readlines()
    except FileNotFoundError:
        print('Error: could not open schedule file: {}'.format(schedule_file_name))

    # remove header if exists
    try:
        int(lines[0].split(';')[0])
        header = 0
    except ValueError:
        lines = lines[1:]
        header = 1

    # create schedule list
    schedule = [list(map(int, line.rstrip('\n,;').split(';'))) for line in lines]

    # ############## validation ##########################
    # iteration over each schedule-line
    for i in range(0, len(schedule)):

        pos = schedule[i]

        # check number of elements per line
        if len(pos) != 4:
            print('Error line {} ({}): invalid number of elements per line'.format(i + 1 + header, pos))
            exit()

        vehicle_id = pos[0]
        location = pos[1]
        unload = pos[2]
        load = pos[3]

        # check syntax
        if not isinstance(location, int) or location > 6:
            print('Error line {} ({}): undefined location'.format(i+1+header, pos))
            exit()
        if not isinstance(vehicle_id, int):
            print('Error line {} ({}): undefined vehicle ID'.format(i+1+header, pos))
            exit()
        if unload not in (0, 1):
            print('Error line {} ({}): undefined unload command'.format(i+1+header, pos))
            exit()
        if load not in (0, 1):
            print('Error line {} ({}): undefined load command'.format(i+1+header, pos))
            exit()

        # get previous line
        if i > 0:
            last_pos = schedule[i - 1]
            last_vehicle_id = last_pos[0]
            last_location = last_pos[1]
            last_load = last_pos[3]

            if location == last_location and vehicle_id == last_vehicle_id:
                print('Error line {} ({}): a vehicle cannot visit the same location twice in a row'.format(i + 1 + header, pos))
                exit()

        else:
            if unload == 1:
                print('Error line {} ({}): a vehicle cannot unload at its start location'.format(i+1+header, pos))
                exit()
            continue

        # get next line
        try:
            next_pos = schedule[i + 1]
            next_vehicle_id = next_pos[0]
            next_unload = next_pos[2]

            if not last_vehicle_id <= vehicle_id <= next_vehicle_id:
                print('Error line {} ({}): schedule has to be ordered by vehicle ID'.format(i + 1+header, pos))
                exit()
        except IndexError:
            if load == 1:
                print('Error line {} ({}): vehicle cannot load at last location'.format(i+1+header, pos))
                exit()

        # check start position
        if i == 0 or vehicle_id != last_vehicle_id:
            if location != vehicle_id:
                print('Error line {} ({}): each vehicle has to start at its corresponding loadpoint'.format(i + 1 + header, pos))
                exit()

        # check load-unload order
        if unload == 1:
            if last_vehicle_id != vehicle_id:
                print('Error line {} ({}): vehicle cannot unload at its start location'.format(i+1+header, pos))
                exit()
            if last_load == 0:
                print('Error line {} ({}): empty vehicle cannot unload'.format(i+1+header, pos))
                exit()

            # correct transport; lower demand
            demand[last_location][location] -= 1

        # check load-unload order
        if load == 1:
            if next_vehicle_id == vehicle_id and next_unload == 0:
                print('Error line {} ({}): full vehicle has to unload at next station'.format(i+1+header, pos))
                exit()
            if next_vehicle_id != vehicle_id:
                print('Error line {} ({}): vehicle cannot load at last location'.format(i+1+header, pos))
                exit()

    # check a valid solution for demand fulfillment
    valid = True
    for start, rest in demand.items():
        for dest, number in rest.items():
            if number > 0:
                print('Invalid solution: not all transports from {} to {} done'.format(start, dest))
                valid = False
            elif number < 0:
                print('Invalid solution: too many transports from {} to {} done'.format(start, dest))
                valid = False

    if valid:
        # ############## score for valid solution ############################

        # caluclate distances between machines
        distance = dict()

        def euclidean_distance(p1, p2):
            return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

        for start in range(0, len(machine_positions)):
            try:
                distance[start]
            except KeyError:
                distance[start] = dict()
            for dest in range(0, len(machine_positions)):
                distance[start][dest] = euclidean_distance(machine_positions[start], machine_positions[dest])

        # calculate the distances of all vehicles
        vehicle_distances = dict()
        for i in range(1, len(schedule)):
            pos = schedule[i]
            vehicle_id = pos[0]
            location = pos[1]

            last_pos = schedule[i - 1]
            last_vehicle_id = last_pos[0]
            last_location = last_pos[1]

            if vehicle_id != last_vehicle_id:
                continue
            else:
                d = distance[last_location-1][location-1]
                try:
                    vehicle_distances[vehicle_id] += d
                except KeyError:
                    vehicle_distances[vehicle_id] = d

        # calculate the time for all vehicle distances
        vehicle_times = [dist/velocity for dist in vehicle_distances.values()]

        score = max(vehicle_times)
        print('Valid solution; Score: {}'.format(round(score, 4)))

    return True


if __name__ == "__main__":
    try:
        validation('transport_demand.txt', 'schedule.txt')
    except SystemExit:
        pass
    except Exception as e:
        print('unexpected error')
