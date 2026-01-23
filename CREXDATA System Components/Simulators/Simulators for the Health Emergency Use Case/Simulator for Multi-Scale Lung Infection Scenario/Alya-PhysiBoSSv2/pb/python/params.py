import xml.etree.ElementTree as ET
import json
import os
from sys import argv

def update_param(root, param_name, param_value):
    """
    Updates an XML parameter by searching for the correct path.
    """
    xpath = param_name.replace(".", "/")  # Convert JSON-style keys to XML path

    elements = root.findall(f".//{xpath}")
    if elements:
        for el in elements:
            el.text = str(param_value)
        print(f"Updated: {param_name} -> {param_value}")
    else:
        print(f"Warning: Parameter '{param_name}' not found in XML")

def params_to_xml(json_data, default_xml, output_xml,instance_num,turbine):
    """
    Reads an XML template, updates it using JSON parameters, and writes the modified XML to a new file.
    """
    if not os.path.exists(default_xml):
        print(f"Error: Default XML file '{default_xml}' not found.")
        return

    print(json_data)
    tree = ET.parse(default_xml)
    root = tree.getroot()

    for param, value in json_data.items():
        update_param(root, param, value)

    # Also update the instance folder
    update_param(root, "save.folder", instance_num)
    update_param(root,"user_parameters.turbine",turbine)
    tree.write(output_xml)
    

    try:
        with open(os.path.join(instance_num, "variables.txt"), "w") as f:
            val = (
                str(json_data["user_parameters.particle_type"])
                +" " +str(json_data["user_parameters.alveoli_id"])
                +" " + str(json_data["user_parameters.multiplicity_of_infection"])
            )
            f.write(val + "\n")
    except KeyError as e:
        print(f"Missing key in JSON: {e}")
    try:
        with open(os.path.join(instance_num, "moi.txt"), "w") as f:
            val = (str(json_data["user_parameters.multiplicity_of_infection"])
            )
            f.write(val + "\n")
    except KeyError as e:
        print(f"Missing key in JSON: {e}")
if __name__ == '__main__':
    if len(argv) < 5:
        print("Usage: script.py '<json_string>' default.xml output.xml instance_num turbine")
        exit(1)

    params = json.loads(argv[1])
    default_xml = argv[2]
    output_xml = argv[3]
    instance_num = argv[4]
    turbine = argv[5]

    params_to_xml(params, default_xml, output_xml, instance_num,turbine)
