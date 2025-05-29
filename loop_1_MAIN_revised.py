#import os
import heapq
import time

import traci
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
#import traci.constants as tc
import matplotlib.pyplot as plt
import sys
#import numpy as np
#import math
#import seaborn as sns
import random
import os
import sumolib
from collections import defaultdict
from tabulate import tabulate
from DQN2 import *

#python loop_1_MAIN.py loop1.sumocfg ## Run this command in the terminal to execute the script
# can be used multiple simulations at the same time
# Define the filename for the saved state
STATE_FILENAME = "SAVE_state.xml.gz"  # Replace with your actual saved state file
# Check for command-line argument
if len(sys.argv) < 2:
    print("Usage: python script.py <sumo_config_file>")
    sys.exit(1)

sumo_config_file = sys.argv[1]
# Connect to the SUMO server


# Constants
BATTERY_CAPACITY = 640 # battery capacity of electric vehicles in Wh
BATTERY_CAPACITY_POD = 2000 # battery capacity of charging pods in Wh
LOW_BATTERY_THRESHOLD = 30 # 20% battery capacity
SLOWDOWN_SPEED_ZERO_RANGE = 3 # reduced speed for vehicles with zero remaining range
SLOWDOWN_SPEED_LOW_BATTERY = 6.5 # reduced speed for vehicles with low battery
WIRELESS_POD_POWER_RATING = 18000  # W
CHARGE_RATE = WIRELESS_POD_POWER_RATING / 3600  # Wh per second
DURATION = 80  # seconds
total_energy_charged = 0
elec_consumption = 0
total_energy_delivered = 0
total_energy_delivered_ini = 0
CHARGING_DISTANCE_THRESHOLD = 40  # meters
Max_charge_for_EVs= 80
parking_end = 480
edge_end=850
PARKING_AREA_CONPACITY = 6
warm_up_time = 3000

# The number of parking lot
#parking_areas_list = [f"pa_{i}" for i in range(12)]

parking_areas_list = [f"pa_{i}" for i in range(4)]

delayed_parking_pods = []
parking_area_reservations = {pa: 0 for pa in parking_areas_list}

from collections import defaultdict
import heapq
import traci
import sumolib


def optimized_proximity_map(net_file, max_neighbors=2):
    """
    Build proximity map with downstream parking areas as keys and upstream sources as values

    Args:
        net_file (str): Path to SUMO network file
        max_neighbors (int): Maximum number of upstream neighbors to store per downstream

    Returns:
        dict: {downstream_pa: [(upstream_pa1, distance1), (upstream_pa2, distance2)]}
    """
    proximity_map = defaultdict(list)
    net = sumolib.net.readNet(net_file)
    parking_areas = parking_areas_list

    # Build a temporary forward map first
    forward_map = defaultdict(list)

    for src_pa in parking_areas:
        src_edge = traci.parkingarea.getLaneID(src_pa).split("_")[0]
        temp_distances = []

        for dst_pa in parking_areas:
            if src_pa == dst_pa:
                continue

            try:
                route = traci.simulation.findRoute(
                    src_edge,
                    traci.parkingarea.getLaneID(dst_pa).split("_")[0]
                )
                if route.length > 0:
                    heapq.heappush(temp_distances, (route.length, dst_pa))
            except traci.TraCIException:
                continue

        forward_map[src_pa] = [
            heapq.heappop(temp_distances)
            for _ in range(min(max_neighbors, len(temp_distances)))
        ]

    # Reverse the mapping
    for src_pa, neighbors in forward_map.items():
        for distance, dst_pa in neighbors:
            proximity_map[dst_pa].append((src_pa, distance))

    # Sort each downstream entry by distance
    for dst_pa in proximity_map:
        proximity_map[dst_pa].sort(key=lambda x: x[1])

    return proximity_map


def find_eligible_evs_for_pod(pod_id, parking_areas, target_pa, threshold=3):
    """
    Find eligible EVs that:
    1. Are on the same edge as the pod
    2. Have battery level below threshold
    3. Have route passing through at least one underutilized parking area

    Args:
        pod_id (str): ID of the charging pod
        parking_areas (list): List of all parking area IDs
        threshold (int): Minimum vehicles needed at parking area

    Returns:
        list: Eligible EV IDs sorted by battery level (lowest first)
    """
    pod_edge = traci.vehicle.getRoadID(pod_id)
    eligible_evs = []

    for ev_id in traci.vehicle.getIDList():
        if ev_id not in assigned_charging_pod_for_electric_veh:
            # Filter for electric vehicles
            if traci.vehicle.getTypeID(ev_id) != "ElectricVehicle":
                continue

            # Check if on same edge as pod
            if traci.vehicle.getRoadID(ev_id) != pod_edge:
                continue

            # Check battery level
            battery_level = (float(
                traci.vehicle.getParameter(ev_id, "device.battery.actualBatteryCapacity")) / BATTERY_CAPACITY) * 100
            if battery_level > 80:
                continue

            # Check if route passes through any underutilized parking area
            ev_route = traci.vehicle.getRoute(ev_id)
            pa_lane = traci.parkingarea.getLaneID(target_pa)
            pa_edge = pa_lane.split('_')[0]
            veh_pos = traci.vehicle.getPosition(ev_id)
            pod_pos = traci.vehicle.getPosition(pod_id)
            distance2D = traci.simulation.getDistance2D(veh_pos[0], veh_pos[1], pod_pos[0], pod_pos[1])
            if pa_edge in ev_route[-1]:
                eligible_evs.append((ev_id, distance2D))

    # Sort by distance2D (lowest first)
    return [ev_id for ev_id, _ in sorted(eligible_evs, key=lambda x: x[1])]


def smart_pod_redistribution(threshold=2, proximity_map=None):
    parking_areas = parking_areas_list

    # Find underutilized parking areas
    underutilized = [pa for pa in parking_areas
                     if traci.parkingarea.getVehicleCount(pa) < threshold]

    for target_pa in underutilized:
        # Find nearest available pod
        for source_pa, _ in proximity_map.get(target_pa, []):
            if source_pa not in underutilized:
                parked_pods = [v for v in traci.parkingarea.getVehicleIDs(source_pa)
                               if v not in assigned_charging_pod_for_electric_veh]

                for pod_id in parked_pods:
                    if pod_id not in assigned_charging_pod_for_electric_veh:
                        battery_capacity_percentage_pod = (float(traci.vehicle.getParameter(pod_id,
                                                                                            "device.battery.actualBatteryCapacity")) / BATTERY_CAPACITY_POD) * 100
                        if battery_capacity_percentage_pod < ((BATTERY_CAPACITY*2)/BATTERY_CAPACITY_POD)*100:
                            continue
                        # Find eligible EVs for this pod
                        eligible_evs = find_eligible_evs_for_pod(pod_id, parking_areas, target_pa, threshold)

                        if eligible_evs:
                            # Move pod to target parking area
                            handle_low_battery_vehicle(eligible_evs[0])
                            print(f"pod {pod_id} is redistributed for EV {eligible_evs[0]}")
                            return


def get_state(vehicle_id=None, charging_pod_id=None):
    state = []
    # Pod info
    pod_battery = float(traci.vehicle.getParameter(charging_pod_id, "device.battery.actualBatteryCapacity"))
    SOC_POD = (pod_battery / BATTERY_CAPACITY_POD) * 100
    pod_lane_pos = traci.vehicle.getLanePosition(charging_pod_id)
    pod_edge = traci.vehicle.getRoadID(charging_pod_id)
    if vehicle_id is None:
        default_state = np.array([
            -1.0,  # EV_SOC (-1 when unknown)
            SOC_POD,  # Pod_SOC
            -1.0,  # Max distance (unknown position)
            -9999.0,  # lane distance (unknown)
            0.0  # Not charging
        ], dtype=np.float32)
        return default_state

    else:
        # EV info
        battery = float(traci.vehicle.getParameter(vehicle_id, "device.battery.actualBatteryCapacity"))
        SOC_EV = (battery / BATTERY_CAPACITY) * 100
        lane_pos = traci.vehicle.getLanePosition(vehicle_id)
        veh_edge = traci.vehicle.getRoadID(vehicle_id)

        # distance
        veh_pos = traci.vehicle.getPosition(vehicle_id)
        pod_pos = traci.vehicle.getPosition(charging_pod_id)
        distance2D = traci.simulation.getDistance2D(veh_pos[0], veh_pos[1], pod_pos[0], pod_pos[1])

        if 'J' not in veh_edge and 'J' not in pod_edge:  # Both EV and Pod are not at junction
            if pod_lane_pos < 0:
                lane_distance = 500 - lane_pos
            else:
                lane_distance = pod_lane_pos - lane_pos
        else:
            lane_distance = -distance2D

        if lane_pos < 50 and pod_lane_pos > 950:
            lane_distance = -distance2D

        # Check if the EV and Pod are currently charging
        is_charging = False
        if (vehicle_id, charging_pod_id) in charging_pairs:
            is_charging = charging_pairs[(vehicle_id, charging_pod_id)]

        is_assigned = False
        if charging_pod_id in assigned_charging_pod_for_electric_veh.values():
            is_assigned = True

        state = [
            SOC_EV,
            SOC_POD,
            distance2D,
            lane_distance,  # negative when the POD is behind EV
            int(is_charging)  # 1 if charging, 0 otherwise
        ]

        return np.array(state, dtype=np.float32)


def calculate_average_ev_soc():
    active_vehicles = traci.vehicle.getIDList()
    ev_soc_values = []

    for vehicle_id in active_vehicles:
        vehicle_type = traci.vehicle.getTypeID(vehicle_id)

        if vehicle_type == "ElectricVehicle":
            try:
                # 获取电池容量信息
                battery = float(traci.vehicle.getParameter(vehicle_id, "device.battery.actualBatteryCapacity"))
                soc = (battery / BATTERY_CAPACITY) * 100  # 计算SOC百分比
                ev_soc_values.append(soc)
            except traci.exceptions.TraCIException as e:
                print(f"Error getting battery info for {vehicle_id}: {e}")
                continue


    if len(ev_soc_values) > 0:
        average_soc = sum(ev_soc_values) / len(ev_soc_values)
        return round(average_soc, 2)
    else:
        return 0.0


def calculate_average_pod_soc():
    active_vehicles = traci.vehicle.getIDList()
    ev_soc_values = []

    for vehicle_id in active_vehicles:
        vehicle_type = traci.vehicle.getTypeID(vehicle_id)

        if vehicle_type == "ChargingPod":
            try:
                # 获取电池容量信息
                battery = float(traci.vehicle.getParameter(vehicle_id, "device.battery.actualBatteryCapacity"))
                soc = (battery / BATTERY_CAPACITY) * 100
                ev_soc_values.append(soc)
            except traci.exceptions.TraCIException as e:
                print(f"Error getting battery info for {vehicle_id}: {e}")
                continue

    if len(ev_soc_values) > 0:
        average_soc = sum(ev_soc_values) / len(ev_soc_values)
        return round(average_soc, 2)
    else:
        return 0.0


ACTION_SPACE = [
    "no_action",
    "stop_charging",
    "start_charging"
]


def calculate_reward(vehicle_id, charging_pod_id, action):
    reward = 0
    state = get_state(vehicle_id, charging_pod_id)
    # Get vehicle and pod information
    SOC_EV = state[0]
    SOC_POD = state[1]
    isCharging = state[4]
    distance2D = state[2]
    lane_distance = state[3]
    average_ev_soc = calculate_average_ev_soc()
    average_pod_soc = calculate_average_pod_soc()
    if traci.vehicle.getLanePosition(charging_pod_id) < 0:
        distance2Park = 0
    else:
        distance2Park = find_nearest_parking_distance(charging_pod_id, parking_areas_list)

    # --- Reward rules ---
    if action == "no_action":
        if distance2D < CHARGING_DISTANCE_THRESHOLD and lane_distance < 0:
            reward -= 100
            print(f"{vehicle_id}: {SOC_EV} and {charging_pod_id}: {SOC_POD} should choose start_charging, distance is {lane_distance}")

    if action == "start_charging":
        if distance2D < CHARGING_DISTANCE_THRESHOLD and lane_distance < 0:
            if distance2Park < 50:
                # Pod should park now
                if SOC_EV >= 60 or SOC_POD < LOW_BATTERY_THRESHOLD:
                    reward -= 1
                # Can't stop now
                else:
                    reward += 2
            else:
                reward += 1
        else:
            reward -= 10

    elif action == "stop_charging":
        if distance2Park < 50:
            # Pod should park now
            if SOC_EV >= 60 or SOC_POD < LOW_BATTERY_THRESHOLD:
                reward += 100
            # Can't stop now
            else:
                reward -= 10
        else:
            if SOC_POD < LOW_BATTERY_THRESHOLD:
                reward += 20
            else:
                reward -= 10
        '''
        if SOC_EV >= Max_charge_for_EVs or SOC_POD <= LOW_BATTERY_THRESHOLD:
            reward += 30 - 0.07 * distance2Park
        elif SOC_EV >= 60 and distance2Park < 70:
            reward += 30 - 0.07 * distance2Park
        else:
            #print(f"{charging_pod_id} should not park at distance {distance2Park} since didn't charge properly")
            reward -= 15
        '''
    reward += (SOC_EV - 20) * 0.05
    reward -= (100 - SOC_POD) * 0.03
    return reward


# Initialize sets to keep track of counted vehicles and a dictionary to keep track of assigned t_1 vehicles for each electric bus
zero_range_vehicles = set()
low_battery_vehicles = set()
assigned_charging_pod_for_electric_veh = {}
# Initialize a dictionary to keep track of the queues for each parking area
#parking_area_queues = {parking_area_id: set() for parking_area_id in traci.parkingarea.getIDList()}

# Lists to store timestamps and speeds of electric bus vehicles
timestamps = []
speeds = []

# Dictionary to track the last speed and time for each charging pod
charging_pod_speeds = {}

def handle_electric_vehicle(vehicle_id, distance, total_energy_consumed, actual_battery_capacity, electric_veh_lane):
    """
    Handle electric vehicle behavior.

    Args:
        vehicle_id (str): The ID of the electric vehicle.
        distance (float): Distance traveled by the vehicle.
        total_energy_consumed (float): Total energy consumed by the vehicle.
        actual_battery_capacity (float): Actual battery capacity of the vehicle.
        electric_veh_lane (str): Lane ID of the electric vehicle.

    Returns:
        None
    """
    global LOW_BATTERY_THRESHOLD
    global zero_range_vehicles
    global assigned_charging_pod_for_electric_veh
    if total_energy_consumed > 0:
        mWh = distance / total_energy_consumed
        remaining_range = actual_battery_capacity * mWh
        battery_capacity_percentage = (actual_battery_capacity / BATTERY_CAPACITY) * 100

        if remaining_range == 0 and battery_capacity_percentage==0 and vehicle_id not in zero_range_vehicles:
            edge_id = traci.vehicle.getRoadID(vehicle_id)
            print(f"Electric Veh {vehicle_id} on edge {edge_id} has zero remaining range at time {traci.simulation.getTime()}")
            zero_range_vehicles.add(vehicle_id)# Add vehicle to the set of vehicles with zero remaining range
            traci.vehicle.slowDown(vehicle_id, SLOWDOWN_SPEED_ZERO_RANGE, duration=0) # Reduce speed to 3 m/s
            traci.vehicle.setColor(vehicle_id, (255, 0, 0))  # Red color


        if battery_capacity_percentage < LOW_BATTERY_THRESHOLD and traci.vehicle.getLanePosition(vehicle_id) < edge_end:#and traci.vehicle.getLanePosition(vehicle_id) >460 and traci.vehicle.getLanePosition(vehicle_id) < 530:#
            handle_low_battery_vehicle(vehicle_id)
        elif battery_capacity_percentage >= LOW_BATTERY_THRESHOLD and vehicle_id not in assigned_charging_pod_for_electric_veh:
            traci.vehicle.setColor(vehicle_id, (255, 255, 255)) #white color
            traci.vehicle.changeLane(vehicle_id, 0, duration=80) # keep vehicle in the right lane with high battery


def handle_low_battery_vehicle(vehicle_id):
    """
    Handle electric vehicle with low battery.

    Args:
        vehicle_id (str): The ID of the electric vehicle.
        electric_veh_lane (str): Lane ID of the electric vehicle.

    Returns:
        None
    """
    global low_battery_vehicles
    low_battery_vehicles.add(vehicle_id)
    traci.vehicle.slowDown(vehicle_id, SLOWDOWN_SPEED_LOW_BATTERY, duration=10)
    traci.vehicle.changeLane(vehicle_id, 1, duration=9999)
    traci.vehicle.setColor(vehicle_id, (255, 0, 0))  # Red color
    global assigned_charging_pod_for_electric_veh
    edge_id = traci.vehicle.getRoadID(vehicle_id)
    nearest_charging_pod_id = None
    min_distance = float('inf')
    electric_veh_position = traci.vehicle.getPosition(vehicle_id)
    electric_veh_lane_position = traci.vehicle.getLanePosition(vehicle_id)
    battery_capacity_percentage = (float(traci.vehicle.getParameter(vehicle_id, "device.battery.actualBatteryCapacity")) / BATTERY_CAPACITY) * 100

    if vehicle_id in traci.vehicle.getIDList():
        for charging_pod_id in traci.vehicle.getIDList():
            battery_capacity_percentage_pod = (float(traci.vehicle.getParameter(charging_pod_id,
                                                                                "device.battery.actualBatteryCapacity")) / BATTERY_CAPACITY_POD) * 100
            if (traci.vehicle.getTypeID(charging_pod_id) == "ChargingPod" and traci.vehicle.getRoadID(
                    charging_pod_id) == edge_id and traci.vehicle.isStoppedParking(charging_pod_id) and vehicle_id not in assigned_charging_pod_for_electric_veh and battery_capacity_percentage_pod > ((BATTERY_CAPACITY*2)/BATTERY_CAPACITY_POD)*100): # check if the pod has enough energy
                charging_pod_position = traci.vehicle.getPosition(charging_pod_id)
                #charging_pod_lane_position = traci.vehicle.getLanePosition(charging_pod_id)
                distance_to_electric_veh = traci.simulation.getDistance2D(electric_veh_position[0],
                                                                          electric_veh_position[1],
                                                                          charging_pod_position[0],
                                                                          charging_pod_position[1])

                if distance_to_electric_veh < min_distance and charging_pod_id not in assigned_charging_pod_for_electric_veh.values() and electric_veh_lane_position > parking_end:
                    nearest_charging_pod_id = charging_pod_id
                    min_distance = distance_to_electric_veh

        if nearest_charging_pod_id:
            assigned_charging_pod_for_electric_veh[vehicle_id] = nearest_charging_pod_id
            print(f"Electric Veh {vehicle_id} assigned to Charging Pod {nearest_charging_pod_id} at time {traci.simulation.getTime()}")


def share_energy(charging_pod_id, vehicle_id):
    """
    Share energy between a charging pod and an electric vehicle.

    Args:
        charging_pod_id (str): The ID of the charging pod.
        vehicle_id (str): The ID of the electric vehicle.
    Returns:
        None
    """
    global total_energy_charged
    global warm_up_time
    actual_battery_capacity_pod = float(
        traci.vehicle.getParameter(charging_pod_id, "device.battery.actualBatteryCapacity"))
    #elec_consumption = float(traci.vehicle.getElectricityConsumption(charging_pod_id))
    new_energy_charging_pod = actual_battery_capacity_pod - CHARGE_RATE
    actual_battery_capacity = float(traci.vehicle.getParameter(vehicle_id, "device.battery.actualBatteryCapacity"))
    new_energy_electric = actual_battery_capacity + CHARGE_RATE
    traci.vehicle.setParameter(charging_pod_id, "device.battery.actualBatteryCapacity", new_energy_charging_pod)
    traci.vehicle.setParameter(vehicle_id, "device.battery.actualBatteryCapacity", new_energy_electric)
    if traci.simulation.getTime() >= warm_up_time: #Steady state starts
        total_energy_charged += CHARGE_RATE
    #print(f"Energy shared: {new_energy_electric} kWh")


def simulate_step():
    """
    Perform a single simulation step.
    """
    traci.simulationStep()
    timestamps.append(traci.simulation.getTime())


def park_charging_pod_during_warmup(charging_pod_id, parking_areas, duration):
    """
    Park the charging pod at an allocated parking area during warmup.

    para:
        charging_pod_id (str): The ID of the charging pod, e.g., 'p_0.0', 'p_0.1'
        parking_areas (list): List of parking areas, e.g., ['pa_0', 'pa_1', ..., 'pa_11']
        duration (float): Duration to park the charging pod
    """
    # Get the true pod number from the charging pod ID
    pod_number = int(charging_pod_id.split('.')[-1])

    # Allocate parking evenly based on pod number
    parking_index = pod_number % len(parking_areas)
    parking_area = parking_areas[parking_index]

    # Get parking area information (if needed for other actions)
    # For example, you might get the lane ID and other details if required
    lane_id = traci.parkingarea.getLaneID(parking_area)
    edge_id = lane_id.split('_')[0]  # Extract edge ID from the lane ID

    # Change the target of the charging pod to the edge of the parking area
    traci.vehicle.changeTarget(charging_pod_id, edge_id)

    traci.vehicle.setParameter(charging_pod_id, "parking.friendlyPos", "true")

    # Set the vehicle to stop at the specified parking area
    traci.vehicle.setParkingAreaStop(charging_pod_id, parking_area, duration)


def find_nearest_parking_area(pod_id, parking_areas_list):
    """
    Find the nearest reachable parking area for the charging pod based on current vehicle occupancy.

    :param pod_id: ID of the charging pod (string).
    :param parking_areas_list: List of parking area IDs (list of strings).
    :return: ID of the nearest reachable parking area (string). Returns None if no suitable parking area is found.
    """
    # Get current lane and position of the pod
    current_lane = traci.vehicle.getLaneID(pod_id)
    current_position = traci.vehicle.getLanePosition(pod_id)
    current_road = current_lane.split('_')[0]  # Extract road ID from lane ID

    # Handle junction case
    if "J" in current_road:
        outgoing_edges = traci.junction.getOutgoingEdges(current_road[1:])
        cur_route = traci.vehicle.getRoute(pod_id)
        for road in reversed(cur_route):
            if road in outgoing_edges:
                current_road = road
                break
        current_position = 0

    min_distance = float('inf')
    nearest_parking_area = None

    for parking_area in parking_areas_list:
        # Get current vehicles in the parking area (ignore reservations)
        current_vehicles = traci.parkingarea.getVehicleCount(parking_area)
        available = PARKING_AREA_CONPACITY - current_vehicles

        # Skip if parking area is full
        if available <= 0:
            continue

        parking_area_lane = traci.parkingarea.getLaneID(parking_area)
        parking_area_road = parking_area_lane.split('_')[0]

        # Case 1: Parking area is on the same road
        if current_road == parking_area_road:
            parking_area_position = traci.parkingarea.getEndPos(parking_area)
            if current_position - parking_area_position < 0:
                distance = abs(current_position - parking_area_position)
            else:
                continue  # Pod is ahead of the parking area

        # Case 2: Parking area is on a different road
        else:
            route = traci.simulation.findRoute(current_road, parking_area_road)
            if not route or not route.edges:
                continue  # No valid route

            # Calculate total driving distance
            total_distance = traci.lane.getLength(current_lane) - current_position
            for edge in route.edges[1:-1]:
                lane_id = f"{edge}_0"
                total_distance += traci.lane.getLength(lane_id)
            total_distance += traci.parkingarea.getEndPos(parking_area)
            distance = total_distance

        # Update nearest parking area if closer
        if distance < min_distance:
            min_distance = distance
            nearest_parking_area = parking_area

    return  min_distance, nearest_parking_area


def find_nearest_parking_distance(pod_id, parking_areas_list):
    """
    Find the distance to the nearest available parking area for the charging pod.

    :param pod_id: The ID of the charging pod (string).
    :param parking_areas_list: A list of parking area IDs (list of strings).
    :return: The distance to the nearest available parking area (float).
             Returns None if no available parking area is found.
    """

    # Get the current lane ID and position of the pod
    current_lane = traci.vehicle.getLaneID(pod_id)
    current_position = traci.vehicle.getLanePosition(pod_id)
    current_road = current_lane.split('_')[0]  # Extract road ID from lane ID

    # If the pod is on the junction
    if "J" in current_road:
        outgoing_edges = traci.junction.getOutgoingEdges(current_road[1:])
        cur_route = traci.vehicle.getRoute(pod_id)
        for road in reversed(cur_route):
            if road in outgoing_edges:
                current_road = road
                break
        current_position = 0

    # Initialize minimum distance
    min_distance = float('inf')  # Set to infinity initially
    cur_route = None
    parking_area = ""

    # Iterate through all parking areas
    for parking_area in parking_areas_list:
        # Check if parking area has available space
        current_vehicles = traci.parkingarea.getVehicleCount(parking_area)
        if current_vehicles >= PARKING_AREA_CONPACITY:
            continue

        # Get the lane ID of the parking area
        parking_area_lane = traci.parkingarea.getLaneID(parking_area)
        parking_area_road = parking_area_lane.split('_')[0]  # Extract road ID from lane ID

        # Check if the parking area is on the same road as the pod
        if current_road == parking_area_road:
            # Calculate the distance between the pod and the parking area
            parking_area_position = traci.parkingarea.getEndPos(parking_area)
            # Only if the pod is behind the parking area, it can be recorded
            if current_position - parking_area_position < 0:
                distance = abs(current_position - parking_area_position)
            else:
                continue
        else:
            # Check if there is a valid route from the current road to the parking area road
            route = traci.simulation.findRoute(current_road, parking_area_road)

            if route and route.edges:  # If a valid route exists
                # Calculate the total driving distance along the route
                total_distance = 0.0

                # Add the remaining distance on the current lane
                current_lane_length = traci.lane.getLength(current_lane)
                total_distance += (current_lane_length - current_position)

                # Add the length of all intermediate edges in the route
                for edge in route.edges[1:-1]:  # Skip first and last edges
                    lane_id = f"{edge}_0"
                    total_distance += traci.lane.getLength(lane_id)

                # Add the partial distance on the last edge
                parking_area_position = traci.parkingarea.getEndPos(parking_area)
                total_distance += parking_area_position

                distance = total_distance
                cur_route = route
            else:
                # No valid route found, skip this parking area
                continue

        # Update minimum distance if this parking area is closer
        if distance < min_distance:
            min_distance = distance
            best_route = parking_area

    # Return the minimum distance found (or None if no available parking)
    return min_distance if min_distance != float('inf') else None

def vehicle_exists(vehicle_id):
    try:
        return vehicle_id in traci.vehicle.getIDList()
    except:
        return False


charging_pairs = {}


def main():
    vehicle_data = {}
    charging_pod_data = {}

    global assigned_charging_pod_for_electric_veh
    global parking_area_queues
    global charging_pod_speeds
    cancelled_evs = set()
    # Store vehicles that have already selected a target
    selected_targets = {}

    EPISODES = 30
    MAX_STEPS = 500

    log_file = "training_log_model2_re.txt"
    # This is the format for training log
    col_widths = {
        "Episode": 8,
        "Total Reward": 14,
        "Epsilon": 10,
        "Total loss": 16
    }

    headers = ["Episode", "Total Reward", "Epsilon", "Total loss"]
    with open(log_file, "w") as f:
        header_line = "|".join([h.center(col_widths[h]) for h in headers])
        f.write(f"|{header_line}|\n")
        f.write("-" * (sum(col_widths.values()) + len(headers) + 1) + "\n")  # 分隔线

    # Initialize DQN agent
    state_size = 5  # Based on get_state function
    action_size = len(ACTION_SPACE)
    agent = DQNAgent(state_size, action_size, device='cuda' if torch.cuda.is_available() else 'cpu')

    # Save the model
    best_reward = -float('inf')
    best_model_weights = None
    best_episode = 0

    for episode in range(EPISODES):
        if episode == 110:
            traci.start(["sumo-gui", "-c", sumo_config_file])
        else:
            traci.start(["sumo", "-c", sumo_config_file])
        # Load the saved state
        traci.simulation.loadState("SAVE_state_1x1.xml.gz")
        tatal_loss = 0
        # Restore the data
        total_reward = 0
        assigned_charging_pod_for_electric_veh.clear()
        delayed_parking_pods.clear()
        global parking_area_reservations
        parking_area_reservations = {pa: 0 for pa in parking_areas_list}
        global charging_pairs
        charging_pairs = {}
        print(f"----------------------------------- Below is episode {episode} -----------------------------------")
        for step in range(MAX_STEPS):  # You can adjust the number of steps as needed
            simulate_step()

            # Get the list of all active vehicles
            active_vehicles = traci.vehicle.getIDList()
            for vehicle_id in active_vehicles:
                vehicle_type = traci.vehicle.getTypeID(vehicle_id)

                if vehicle_type == "ElectricVehicle":
                    try:
                        current_edge = traci.vehicle.getRoadID(vehicle_id)
                        # Once the vehicle moves to a new edge, allow it to choose again at the next intersection
                        if vehicle_id in selected_targets and current_edge == selected_targets[vehicle_id]:
                            del selected_targets[vehicle_id]

                        # Change target only if the vehicle is near the intersection (e.g., within 10 meters)
                        if vehicle_id not in selected_targets:
                            if current_edge == "E0":
                                next_edge = "E1"
                            elif current_edge == "E1":
                                next_edge = "E2"
                            elif current_edge == "E2":
                                next_edge = "E3"
                            elif current_edge == "E3":
                                next_edge = "E0"
                            else:
                                continue  # No action for other cases

                            traci.vehicle.changeTarget(vehicle_id, next_edge)
                            selected_targets[vehicle_id] = next_edge  # Store the chosen target

                        distance = traci.vehicle.getDistance(vehicle_id)
                        total_energy_consumed = float(
                            traci.vehicle.getParameter(vehicle_id, "device.battery.totalEnergyConsumed"))
                        actual_battery_capacity = float(
                            traci.vehicle.getParameter(vehicle_id, "device.battery.actualBatteryCapacity"))
                        battery_capacity_percentage = (actual_battery_capacity / BATTERY_CAPACITY) * 100
                        electric_veh_lane = traci.vehicle.getLaneID(vehicle_id)
                    except traci.exceptions.TraCIException as e:
                        print(f"Error handling vehicle {vehicle_id}: {e}")
                        continue
                    handle_electric_vehicle(vehicle_id, distance, total_energy_consumed, actual_battery_capacity,
                                            electric_veh_lane)

                    if vehicle_id not in vehicle_data:
                        vehicle_data[vehicle_id] = {'speed': [], 'battery_capacity': [], 'timestamps': [], 'distance': 0}
                    vehicle_data[vehicle_id]['speed'].append(traci.vehicle.getSpeed(vehicle_id))
                    vehicle_data[vehicle_id]['battery_capacity'].append((actual_battery_capacity / BATTERY_CAPACITY) * 100)
                    vehicle_data[vehicle_id]['timestamps'].append(traci.simulation.getTime())
                    vehicle_data[vehicle_id]['distance'] = distance

                elif vehicle_type == "ChargingPod":
                    # state = traci.vehicle.getStopState(vehicle_id)
                    # print(f"Charging pod {vehicle_id} stop state is {state}.")
                    # Collect data for charging pods
                    try:
                        if traci.simulation.getTime() <= 2900:
                            #freeze the battery
                            traci.vehicle.setParameter(vehicle_id, "device.battery.actualBatteryCapacity",str(BATTERY_CAPACITY_POD))
                            park_charging_pod_during_warmup(vehicle_id, parking_areas_list, 9000)
                        battery_capacity = float(
                            traci.vehicle.getParameter(vehicle_id, "device.battery.actualBatteryCapacity"))
                        charging_pod_data.setdefault(vehicle_id, {'battery_capacity': [], 'timestamps': []})
                        charging_pod_data[vehicle_id]['battery_capacity'].append(
                            (battery_capacity / BATTERY_CAPACITY_POD) * 100)
                        charging_pod_data[vehicle_id]['timestamps'].append(traci.simulation.getTime())

                    except traci.exceptions.TraCIException as e:
                        print(f"Error handling charging pod {vehicle_id}: {e}")
                        continue
                    if vehicle_id in assigned_charging_pod_for_electric_veh.values():
                        charging_pod_id = vehicle_id
                        for v_id, c_id in assigned_charging_pod_for_electric_veh.items():
                            if c_id == charging_pod_id:
                                vehicle_id = v_id
                                break
                        reward = 0
                        if not vehicle_exists(vehicle_id) or not vehicle_exists(charging_pod_id):
                            # We don't hope EV or POD to disappear
                            if vehicle_id in assigned_charging_pod_for_electric_veh:
                                del assigned_charging_pod_for_electric_veh[vehicle_id]
                            print(f"{vehicle_id} or {charging_pod_id} is missing!")
                            reward -= 100
                            continue  # skip

                        current_state = get_state(vehicle_id=vehicle_id, charging_pod_id=charging_pod_id)
                        isCharging = current_state[4]
                        distance2D = current_state[2]
                        lane_distance = current_state[3]
                        SOC_EV = current_state[0]
                        SOC_POD = current_state[1]

                        # Assign the route for eAV and MAP
                        if traci.vehicle.getRoute(vehicle_id)[-1] != traci.vehicle.getRoute(charging_pod_id)[-1]:
                            traci.vehicle.changeTarget(charging_pod_id, traci.vehicle.getRoute(vehicle_id)[-1])

                        if traci.vehicle.isStoppedParking(
                                charging_pod_id) and charging_pod_id in assigned_charging_pod_for_electric_veh.values():
                            traci.vehicle.resume(charging_pod_id)
                            print(
                                f"Charging pod {charging_pod_id} resumed at {traci.simulation.getTime()} with SOC: {SOC_POD}")
                        if traci.vehicle.getLanePosition(charging_pod_id) < 0:
                            distance2Park = 0
                        else:
                            distance2Park = find_nearest_parking_distance(charging_pod_id, parking_areas_list)

                        action_mask = [0] * len(ACTION_SPACE)
                        # Disable invalid actions based on conditions
                        if isCharging:
                            if (SOC_EV > 60 and distance2Park < 50) or SOC_POD < LOW_BATTERY_THRESHOLD:
                                action_mask[ACTION_SPACE.index("stop_charging")] = 1

                        if distance2D < CHARGING_DISTANCE_THRESHOLD and lane_distance < 0:
                            # Create action mask based on current state
                            action_mask[ACTION_SPACE.index("start_charging")] = 1
                        else:
                            action_mask[ACTION_SPACE.index("no_action")] = 1

                        # Get action with masking
                        action_idx = agent.act(current_state, action_mask)
                        action = ACTION_SPACE[action_idx]

                        if action == "no_action":
                            pass
                        elif action == "stop_charging":
                            if SOC_POD <= LOW_BATTERY_THRESHOLD or (SOC_EV >= 60 and distance2Park < 50):
                                # 1. set the color
                                if SOC_EV < LOW_BATTERY_THRESHOLD:
                                    traci.vehicle.setColor(vehicle_id, (255, 0, 0))  # Red color
                                else:
                                    traci.vehicle.setColor(vehicle_id, (255, 255, 255))  # White color
                                    traci.vehicle.changeLane(vehicle_id, 0, duration=80)

                                # 2. Unbind the MAP and the eAV
                                if vehicle_id in assigned_charging_pod_for_electric_veh:
                                    charging_pod_id = assigned_charging_pod_for_electric_veh[vehicle_id]
                                    del assigned_charging_pod_for_electric_veh[vehicle_id]
                                    charging_pairs[(vehicle_id, charging_pod_id)] = False

                                    # This situation means the pod stops charging while just assigned to the ev, which is unwanted
                                    if traci.vehicle.getLanePosition(charging_pod_id) < 0:
                                        print(
                                            f"Pod {charging_pod_id} cannot be parked at this step, waiting for next step")
                                        delayed_parking_pods.append(charging_pod_id)
                                        reward -= 100
                                        continue

                                    # 3.Check if pod is parked
                                    if not traci.vehicle.isStoppedParking(charging_pod_id):
                                        min_distance, nearest_parking = find_nearest_parking_area(charging_pod_id,
                                                                                                  parking_areas_list)
                                        if nearest_parking:
                                            traci.vehicle.changeLane(charging_pod_id, 2, duration=10000)
                                            traci.vehicle.changeTarget(charging_pod_id,
                                                                       traci.parkingarea.getLaneID(
                                                                           nearest_parking).split("_")[0])
                                            traci.vehicle.setParkingAreaStop(charging_pod_id, nearest_parking,
                                                                             duration=90000)
                                            print(
                                                f"Pod {charging_pod_id} for {vehicle_id} will park at {nearest_parking} at distance {min_distance}")
                                            if min_distance > 300:
                                                print(
                                                    f"\033[91mWARNING: {charging_pod_id} is too far away from parking lot distance is {min_distance}\033[0m")
                                        # else:
                                        #  print(f"Pod {charging_pod_id} had parked before")
                                else:
                                    print(
                                        f"\033[91mWARNING: {charging_pod_id}:{SOC_POD} can't stop charging for {vehicle_id}:{SOC_EV} at distance {lane_distance} \033[0m")
                            else:
                                if lane_distance > -5:
                                    traci.vehicle.slowDown(charging_pod_id,
                                                           max(traci.vehicle.getSpeed(vehicle_id) - 2, 0), 0)
                                else:
                                    traci.vehicle.slowDown(charging_pod_id, traci.vehicle.getSpeed(vehicle_id), 0)
                                traci.vehicle.changeLane(charging_pod_id, 1, 10)
                                charging_pairs[(vehicle_id, charging_pod_id)] = False
                        if action == "start_charging":
                            if distance2D < CHARGING_DISTANCE_THRESHOLD and lane_distance < 0:
                                print(
                                    f"{charging_pod_id}(current SOC is {SOC_POD}) is charging {vehicle_id}(current SOC is {SOC_EV}) with a distance {lane_distance} at time {traci.simulation.getTime()}")
                                """
                                else:
                                    print(f"\033[91mWARNING: {charging_pod_id} Shouldn't choose 'start_charging'! "
                                          f"(SOC_EV={SOC_EV}, SOC_POD={SOC_POD}, distance2D={lane_distance})\033[0m")
                                """
                                if lane_distance > -5:
                                    traci.vehicle.slowDown(charging_pod_id,
                                                           max(traci.vehicle.getSpeed(vehicle_id) - 2, 0), 0)
                                else:
                                    traci.vehicle.slowDown(charging_pod_id, traci.vehicle.getSpeed(vehicle_id), 0)
                                share_energy(charging_pod_id, vehicle_id)
                                traci.vehicle.changeLane(charging_pod_id, 1, 10)
                                traci.vehicle.setColor(vehicle_id, (0, 255, 0))  # Green means charging
                                charging_pairs[(vehicle_id, charging_pod_id)] = True
                            else:
                                print(f"\033[91mWARNING: {charging_pod_id} can't choose 'start_charging'! "
                                      f"(SOC_EV={SOC_EV}, SOC_POD={SOC_POD}, distance2D={lane_distance})\033[0m")
                                charging_pairs[(vehicle_id, charging_pod_id)] = False
                        else:
                            if LOW_BATTERY_THRESHOLD < SOC_EV < Max_charge_for_EVs:
                                traci.vehicle.setColor(vehicle_id, (255, 200, 0))

                        next_state = get_state(vehicle_id, charging_pod_id)
                        step_reward = calculate_reward(vehicle_id, charging_pod_id, action) + reward
                        agent.memory.push(current_state, action_idx, step_reward, next_state, False, action_mask)
                        total_reward += step_reward
                        loss = agent.replay()
                    else:
                        charging_pod_id = vehicle_id
                        vehicle_id = None
                        current_state = get_state(vehicle_id=vehicle_id, charging_pod_id=charging_pod_id)
                        isCharging = current_state[4]
                        SOC_POD = current_state[1]

                        action_mask = [0] * len(ACTION_SPACE)  # Start with all actions enabled
                        action_mask[ACTION_SPACE.index("no_action")] = 1
                        # Get action with masking
                        action_idx = agent.act(current_state, action_mask)
                        action = ACTION_SPACE[action_idx]
                        if action == "no_action":
                            pass
                        next_state = get_state(vehicle_id, charging_pod_id)
                        step_reward = 0
                        if not traci.vehicle.isStoppedParking(charging_pod_id) and not isCharging:
                            step_reward = (80 - SOC_POD)*0.02
                        agent.memory.push(current_state, action_idx, step_reward, next_state, False, action_mask)
                        total_reward += step_reward
                        loss = agent.replay()

            # Handle delayed pod
            for delayed_pod_id in list(delayed_parking_pods):
                if not traci.vehicle.isStoppedParking(delayed_pod_id) and traci.vehicle.getLanePosition(delayed_pod_id) >= 0:
                    delayed_parking_pods.remove(delayed_pod_id)
                    min_distance, nearest_parking = find_nearest_parking_area(delayed_pod_id, parking_areas_list)
                    if nearest_parking:
                        traci.vehicle.changeTarget(delayed_pod_id,
                                                   traci.parkingarea.getLaneID(nearest_parking).split("_")[0])
                        traci.vehicle.setParkingAreaStop(delayed_pod_id, nearest_parking, duration=90000)
                        print(f"Pod {delayed_pod_id} parks at {nearest_parking}")
                    else:
                        traci.vehicle.setColor(delayed_pod_id, (255, 165, 0))  #orange colour to mark the pod which cannot be parked
            tatal_loss += loss
        print(f"Episode: {episode}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}, loss: {tatal_loss:.3f}")
        best_reward = total_reward
        best_episode = episode
        os.makedirs("model2_re", exist_ok=True)
        agent.save(f"model2_re/best_model_episode_{episode}.pt")
        print(f"New best model saved with reward {best_reward:.2f}")

        row_data = [
            str(episode).center(col_widths["Episode"]),
            f"{total_reward:.2f}".center(col_widths["Total Reward"]),
            f"{agent.epsilon:.3f}".center(col_widths["Epsilon"]),
            f"{tatal_loss:.3f}".center(col_widths["Total loss"]),
        ]

        with open(log_file, "a") as f:
            f.write(f"|{'|'.join(row_data)}|\n")
        traci.close()


def predict(model_path, sumo_config_file):
    simulation_duration = 7500  # Define the desired simulation duration
    global warm_up_time  # Warm-up time before data collection starts
    global total_energy_charged
    global total_energy_delivered
    global total_energy_delivered_ini
    global elec_consumption
    # Store vehicles that have already selected a target
    selected_targets = {}

    state_size = 5
    action_size = len(ACTION_SPACE)
    agent = DQNAgent(state_size, action_size)

    # Load the model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    agent.load(model_path)
    agent.epsilon = 0.0
    agent.target_model.load_state_dict(agent.model.state_dict())
    agent.model.eval()
    print(f"Loaded model from {model_path}, running in prediction mode.")

    traci.start(["sumo-gui", "-c", sumo_config_file])
    traci.simulation.loadState("SAVE_state_1x1.xml.gz")
    proximity_map = optimized_proximity_map("loop1.net.xml")

    for step in range(simulation_duration):
        simulate_step()
        active_vehicles = traci.vehicle.getIDList()
        for vehicle_id in active_vehicles:
            vehicle_type = traci.vehicle.getTypeID(vehicle_id)

            current_edge = traci.vehicle.getRoadID(vehicle_id)
            # Once the vehicle moves to a new edge, allow it to choose again at the next intersection
            if vehicle_id in selected_targets and current_edge == selected_targets[vehicle_id]:
                del selected_targets[vehicle_id]

            # Change target only if the vehicle is near the intersection (e.g., within 10 meters)
            if vehicle_id not in selected_targets:
                if current_edge == "E0":
                    next_edge = "E1"
                elif current_edge == "E1":
                    next_edge = "E2"
                elif current_edge == "E2":
                    next_edge = "E3"
                elif current_edge == "E3":
                    next_edge = "E0"
                else:
                    continue  # No action for other cases

                traci.vehicle.changeTarget(vehicle_id, next_edge)
                selected_targets[vehicle_id] = next_edge  # Store the chosen target

            if vehicle_type == "ElectricVehicle":
                try:
                    distance = traci.vehicle.getDistance(vehicle_id)
                    total_energy_consumed = float(
                        traci.vehicle.getParameter(vehicle_id, "device.battery.totalEnergyConsumed"))
                    actual_battery_capacity = float(
                        traci.vehicle.getParameter(vehicle_id, "device.battery.actualBatteryCapacity"))
                    lane_id = traci.vehicle.getLaneID(vehicle_id)
                    handle_electric_vehicle(vehicle_id, distance, total_energy_consumed,
                                            actual_battery_capacity, lane_id)
                except traci.exceptions.TraCIException as e:
                    print(f"Error handling vehicle {vehicle_id}: {e}")
                    continue
            elif vehicle_type == "ChargingPod":
                # state = traci.vehicle.getStopState(vehicle_id)
                # print(f"Charging pod {vehicle_id} stop state is {state}.")
                # Collect data for charging pods
                if vehicle_id in assigned_charging_pod_for_electric_veh.values():
                    charging_pod_id = vehicle_id
                    for v_id, c_id in assigned_charging_pod_for_electric_veh.items():
                        if c_id == charging_pod_id:
                            vehicle_id = v_id
                            break

                    if not vehicle_exists(vehicle_id) or not vehicle_exists(charging_pod_id):
                        # We don't hope EV or POD to disappear
                        if vehicle_id in assigned_charging_pod_for_electric_veh:
                            del assigned_charging_pod_for_electric_veh[vehicle_id]
                        print(f"{vehicle_id} or {charging_pod_id} is missing!")
                        continue  # skip

                    current_state = get_state(vehicle_id=vehicle_id, charging_pod_id=charging_pod_id)
                    isCharging = current_state[4]
                    distance2D = current_state[2]
                    lane_distance = current_state[3]
                    SOC_EV = current_state[0]
                    SOC_POD = current_state[1]

                    # Assign the route for eAV and MAP
                    if traci.vehicle.getRoute(vehicle_id)[-1] != traci.vehicle.getRoute(charging_pod_id)[-1]:
                        traci.vehicle.changeTarget(charging_pod_id, traci.vehicle.getRoute(vehicle_id)[-1])

                    if traci.vehicle.isStoppedParking(
                            charging_pod_id) and charging_pod_id in assigned_charging_pod_for_electric_veh.values():
                        traci.vehicle.resume(charging_pod_id)
                        # traci.vehicle.changeLane(charging_pod_id, 2, 0)
                        # traci.vehicle.slowDown(charging_pod_id, 15, duration=0)
                        print(
                            f"Charging pod {charging_pod_id} resumed at {traci.simulation.getTime()} with SOC: {SOC_POD}")

                    action_mask = [0] * len(ACTION_SPACE)  # Start with all actions enabled
                    if traci.vehicle.getLanePosition(charging_pod_id) < 0:
                        distance2Park = 0
                    else:
                        distance2Park = find_nearest_parking_distance(charging_pod_id, parking_areas_list)

                    if isCharging:
                        if (SOC_EV > 60 and distance2Park < 50) or SOC_POD < LOW_BATTERY_THRESHOLD:
                            action_mask[ACTION_SPACE.index("stop_charging")] = 1

                    if distance2D < CHARGING_DISTANCE_THRESHOLD and lane_distance < 0:
                        # Create action mask based on current state
                        action_mask[ACTION_SPACE.index("start_charging")] = 1
                    else:
                        action_mask[ACTION_SPACE.index("no_action")] = 1

                    # Get action with masking
                    action_idx = agent.act(current_state, action_mask)
                    action = ACTION_SPACE[action_idx]
                    if action == "no_action":
                        if lane_distance > 0:
                            traci.vehicle.slowDown(charging_pod_id, 1, 10)
                        else:
                            pass
                    elif action == "stop_charging":
                        if (SOC_EV >= Max_charge_for_EVs or SOC_POD <= LOW_BATTERY_THRESHOLD or (
                                SOC_EV >= 60 and distance2Park < 50)) and traci.vehicle.getLanePosition(charging_pod_id) > 0:
                            # 1. set the color
                            if SOC_EV < LOW_BATTERY_THRESHOLD:
                                traci.vehicle.setColor(vehicle_id, (255, 0, 0))  # Red color
                            else:
                                traci.vehicle.setColor(vehicle_id, (255, 255, 255))  # White color
                                traci.vehicle.changeLane(vehicle_id, 0, duration=80)

                            # 2. Unbind the MAP and the eAV
                            if vehicle_id in assigned_charging_pod_for_electric_veh:
                                charging_pod_id = assigned_charging_pod_for_electric_veh[vehicle_id]
                                del assigned_charging_pod_for_electric_veh[vehicle_id]
                                charging_pairs[(vehicle_id, charging_pod_id)] = False

                                # This situation means the pod stops charging while just assigned to the ev, which is unwanted
                                if traci.vehicle.getLanePosition(charging_pod_id) < 0:
                                    print(
                                        f"Pod {charging_pod_id} cannot be parked at this step, waiting for next step")
                                    delayed_parking_pods.append(charging_pod_id)
                                    continue

                                # 3.Check if pod is parked
                                if not traci.vehicle.isStoppedParking(charging_pod_id):
                                    min_distance, nearest_parking = find_nearest_parking_area(charging_pod_id,
                                                                                              parking_areas_list)
                                    if nearest_parking:
                                        traci.vehicle.changeLane(charging_pod_id, 2, duration=10000)
                                        traci.vehicle.changeTarget(charging_pod_id,
                                                                   traci.parkingarea.getLaneID(
                                                                       nearest_parking).split("_")[0])
                                        traci.vehicle.setParkingAreaStop(charging_pod_id, nearest_parking,
                                                                         duration=90000)
                                        print(
                                            f"Pod {charging_pod_id} for {vehicle_id} will park at {nearest_parking} at distance {min_distance}")
                                        if min_distance > 300:
                                            print(
                                                f"\033[91mWARNING: {charging_pod_id} is too far away from parking lot distance is {min_distance}\033[0m")
                                    # else:
                                    #  print(f"Pod {charging_pod_id} had parked before")
                            else:
                                print(
                                    f"\033[91mWARNING: {charging_pod_id}:{SOC_POD} can't stop charging for {vehicle_id}:{SOC_EV} at distance {lane_distance} \033[0m")
                        else:
                            if lane_distance > -5:
                                traci.vehicle.slowDown(charging_pod_id,
                                                       max(traci.vehicle.getSpeed(vehicle_id) - 2, 0), 0)
                            else:
                                traci.vehicle.slowDown(charging_pod_id, traci.vehicle.getSpeed(vehicle_id), 0)
                            traci.vehicle.changeLane(charging_pod_id, 1, 10)
                            charging_pairs[(vehicle_id, charging_pod_id)] = False
                    if action == "start_charging":
                        if distance2D < CHARGING_DISTANCE_THRESHOLD and lane_distance < 0:
                            print(
                                f"{charging_pod_id} is charging {vehicle_id}(current SOC is {SOC_EV}) with a distance {lane_distance} at time {traci.simulation.getTime()}")
                            """
                            else:
                                print(f"\033[91mWARNING: {charging_pod_id} Shouldn't choose 'start_charging'! "
                                      f"(SOC_EV={SOC_EV}, SOC_POD={SOC_POD}, distance2D={lane_distance})\033[0m")
                            """
                            if lane_distance > -3:
                                traci.vehicle.slowDown(charging_pod_id,
                                                       max(traci.vehicle.getSpeed(vehicle_id) - 2, 0), 0)
                            else:
                                if lane_distance > -5:
                                    traci.vehicle.slowDown(charging_pod_id, traci.vehicle.getSpeed(vehicle_id), 0)
                                else:
                                    traci.vehicle.slowDown(charging_pod_id, traci.vehicle.getSpeed(vehicle_id) + 1, 0)
                            share_energy(charging_pod_id, vehicle_id)
                            traci.vehicle.changeLane(charging_pod_id, 1, 10)
                            traci.vehicle.setColor(vehicle_id, (0, 255, 0))  # Green means charging
                            charging_pairs[(vehicle_id, charging_pod_id)] = True
                        else:
                            print(f"\033[91mWARNING: {charging_pod_id} can't choose 'start_charging'! "
                                  f"(SOC_EV={SOC_EV}, SOC_POD={SOC_POD}, distance2D={lane_distance})\033[0m")
                            charging_pairs[(vehicle_id, charging_pod_id)] = False
                    else:
                        if LOW_BATTERY_THRESHOLD < SOC_EV < Max_charge_for_EVs:
                            traci.vehicle.setColor(vehicle_id, (255, 200, 0))

                        if distance2D > CHARGING_DISTANCE_THRESHOLD:
                            traci.vehicle.changeLane(charging_pod_id, 2, 10)

                else:
                    charging_pod_id = vehicle_id
                    vehicle_id = None
                    current_state = get_state(vehicle_id=vehicle_id, charging_pod_id=charging_pod_id)
                    action_mask = [0] * len(ACTION_SPACE)  # Start with all actions enabled
                    action_mask[ACTION_SPACE.index("no_action")] = 1
                    # Get action with masking
                    action_idx = agent.act(current_state, action_mask)
                    action = ACTION_SPACE[action_idx]
                    if action == "no_action":
                        pass
        if step > 199:
            if step % 10 == 0:
                smart_pod_redistribution(threshold=4, proximity_map=proximity_map)

        # Handle delayed pod
        for delayed_pod_id in list(delayed_parking_pods):
            if not traci.vehicle.isStoppedParking(delayed_pod_id) and traci.vehicle.getLanePosition(
                    delayed_pod_id) >= 0:
                delayed_parking_pods.remove(delayed_pod_id)
                min_distance, nearest_parking = find_nearest_parking_area(delayed_pod_id,
                                                                          parking_areas_list)
                if nearest_parking:
                    traci.vehicle.changeTarget(delayed_pod_id,
                                               traci.parkingarea.getLaneID(nearest_parking).split("_")[0])
                    traci.vehicle.setParkingAreaStop(delayed_pod_id, nearest_parking, duration=90000)
                    print(f"Pod {delayed_pod_id} parks at {nearest_parking}")
                else:
                    traci.vehicle.setColor(delayed_pod_id, (
                    255, 165, 0))  # orange colour to mark the pod which cannot be parked


        ##Calculate energy consumption of pods
        for charging_pod_id in traci.vehicle.getIDList():
            vehicle_type = traci.vehicle.getTypeID(charging_pod_id)
            if traci.simulation.getTime() >= warm_up_time and vehicle_type == "ChargingPod":
                elec_consumptn = float(traci.vehicle.getElectricityConsumption(charging_pod_id))
                elec_consumption += elec_consumptn
                # print(f"Energy consumed by electric vehicles: {elec_consumption} Wh for pod {charging_pod_id}")

        ##Calculate energy charged by charging stations
        for charging_station in traci.chargingstation.getIDList():
            if traci.simulation.getTime() == warm_up_time:  ##time it takes to charge every pod before they are engaged
                energy_charged = float(
                    traci.simulation.getParameter(charging_station, "chargingStation.totalEnergyCharged"))
                total_energy_delivered_ini += energy_charged
                print(f"Energy charged by charging station {charging_station} is {energy_charged} Wh.")
                print(f"Total energy delivered by charging stations: {total_energy_delivered_ini} Wh.")
            elif traci.simulation.getTime() == simulation_duration:
                energy_charged = float(
                    traci.simulation.getParameter(charging_station, "chargingStation.totalEnergyCharged"))
                total_energy_delivered += energy_charged
                print(f"Energy charged by charging station {charging_station} is {energy_charged} Wh.")
                print(f"Total energy delivered by charging stations: {total_energy_delivered} Wh.")

        # Check if the simulation time has exceeded the desired duration
        if traci.simulation.getTime() >= simulation_duration:
            print("Simulation time reached the specified duration. Closing the simulation.")
            break  # Exit the loop and close the simulation

    actual_energy_delivered = total_energy_delivered - total_energy_delivered_ini
    efficiency = total_energy_charged / actual_energy_delivered

    # Print the total count of vehicles with zero remaining range and low battery
    print(f"Total number of vehicles with zero remaining range: {len(zero_range_vehicles)}")
    print(f"Total number of vehicles with less than 25% battery capacity: {len(low_battery_vehicles)}")
    # print(f"Total number of vehicles with cancelled charging: {len(cancelled_evs)} and they are {cancelled_evs}")
    print(f"Total energy charged: {total_energy_charged} Wh")
    print(f"Total energy consumed by electric vehicles: {elec_consumption} Wh")
    print(f"Actual energy delivered by charging stations: {actual_energy_delivered} Wh")
    print(f"Efficiency: {efficiency}")
    traci.close()

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds")
    #predict("model2_re/best_model_episode_29.pt", "test.sumocfg")