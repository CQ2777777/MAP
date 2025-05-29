import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import numpy as np


def prettify(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def generate_electric_vehicle_routes(num_vehicles_per_route, route_ids, route_edges_list, begin, end, min_capacity,
                                     max_capacity,
                                     mean_capacity, std_dev):
    root = ET.Element("routes", xmlns="http://sumo.dlr.de/xsd/routes_file.xsd")

    # 1. 首先定义所有路由
    for route_id, edges in zip(route_ids, route_edges_list):
        route = ET.SubElement(root, "route", id=route_id, edges=edges)

    # 2. 生成车辆 - 每个时间点每个路由都发一辆车
    departure_interval = 10  # 发车间隔(秒)
    total_steps = int((end - begin) / departure_interval) + 1
    vehicle_counter = 0

    for step in range(total_steps):
        depart_time = begin + step * departure_interval

        # 为每个路由在此时刻生成一辆车
        for route_index, route_id in enumerate(route_ids):
            if vehicle_counter >= num_vehicles_per_route * len(route_ids):
                break

            actual_capacity = max(min(np.random.normal(mean_capacity, std_dev), max_capacity), min_capacity)
            vehicle = ET.SubElement(root, "vehicle",
                                    id=f"f_0.{vehicle_counter}",
                                    type="ElectricVehicle",
                                    route=route_id,
                                    depart=str(depart_time))
            param = ET.SubElement(vehicle, "param",
                                  key="device.battery.chargeLevel",
                                  value=str(actual_capacity))
            vehicle_counter += 1

    # 保存文件
    xml_string = prettify(root)
    with open("loop2x2_8ev.rou.xml", "w") as f:
        f.write(xml_string)


# 使用示例
route_ids = ["r_2", "r_3", "r_4", "r_5", "r_6", "r_7", "r_8", "r_9", "r_10", "r_11", "r_12", "r_13"]
route_edges_list = ["E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10", "E11"]

generate_electric_vehicle_routes(
    num_vehicles_per_route=12,
    route_ids=route_ids,
    route_edges_list=route_edges_list,
    begin=3000.00,
    end=4999.00,
    min_capacity=230,
    max_capacity=460,
    mean_capacity=340,
    std_dev=130
)