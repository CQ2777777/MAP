import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import numpy as np

def prettify(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent=" ")


def generate_electric_vehicle_routes(num_vehicles, route_id,  route_edges, begin,  end):
    root = ET.Element("routes", xmlns="http://sumo.dlr.de/xsd/routes_file.xsd")

    # Generate electric vehicle flows with fixed interval depart times and normally distributed battery capacities
    departure_interval = 10  # seconds
    for i in range(num_vehicles):
        depart_time = begin + i * departure_interval
        vehicle = ET.SubElement(root, "vehicle", id=f"p_0.{i}", type="ChargingPod", route=route_id,
                                depart=str(depart_time))
        param = ET.SubElement(vehicle, "param", key="device.battery.chargeLevel", value=str(2000))

    tree = ET.ElementTree(root)
    xml_string = prettify(root)
    with open("Pod_routes.rou.xml", "w") as f:
        f.write(xml_string)


# Example usage
generate_electric_vehicle_routes(
    num_vehicles= 12 * 5,
    route_id="r_t",
    route_edges="E0",
    begin=0.00,
    end=4999.00,

)
