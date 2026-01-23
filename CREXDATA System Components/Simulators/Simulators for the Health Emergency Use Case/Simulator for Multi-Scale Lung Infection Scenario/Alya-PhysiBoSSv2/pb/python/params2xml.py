import xml.etree.ElementTree as ET
import json
import os
from sys import argv
def update_param(root, param_name, param_value):
    """
    Updates an XML parameter by searching for the correct path, including named attributes.
    """
    xpath = param_name.replace(".", "/")  # Convert JSON-style keys to XML path

    # Special handling for microenvironment_setup.variable.oxygen.*
    # Default case: try direct XPath search
    elements = root.findall(f".//{xpath}")
    if elements:
        for el in elements:
            el.text = str(param_value)
        print(f"Updated: {param_name} -> {param_value}")
        return

    print(f"Warning: Parameter '{param_name}' not found in XML")

def params_to_xml(json_data, default_xml, output_xml,instance_num):
    """
    Reads an XML template, updates it using JSON parameters, and writes the modified XML to a new file.
    """
    if not os.path.exists(default_xml):
        print(f"Error: Default XML file '{default_xml}' not found.")
        return
    print(json_data)
    tree = ET.parse(default_xml)
    root = tree.getroot()
    # params = json.loads('%s')
    for param, value in json_data.items():
        update_param(root, param, value)
    update_param(root,"save.folder",instance_num)
    tree.write(output_xml)
    print(json_data["user_parameters.multiplicity_of_infection"])
    return(json_data["user_parameters.multiplicity_of_infection"])

if __name__ == '__main__':
    params = json.loads(argv[1])
    x=params_to_xml(params,argv[2],argv[3],argv[4])
    
