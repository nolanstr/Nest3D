import numpy as np
import pandas as pd
import glob

def read_txt_file(file_path, skiplines=1):
    # Read the TXT file and return the data
    f = open(file_path, "rb")
    lines = f.readlines()
    f.close()
    data = []
    for line in lines[skiplines:]:
        line = [val.decode("ascii") for val in line.split()]
        data.append(line)
    return data

def write_ctf_file(data, output_path, header_info, phase_dict):
    # Define the CTF header
    header = f"""Channel Text File
Prj     Bunge
Author  Converted
JobMode Grid
XCells  {header_info['XCells']}
YCells  {header_info['YCells']}
XStep   {header_info['XStep']}
YStep   {header_info['YStep']}
AcqE1   0.000000
AcqE2   0.000000
AcqE3   0.000000
Euler   Bunge
"""

    # Add phase information to the header
    header += f"Phases  {len(header_info['Phases'])}\n"
    for phase in header_info['Phases']:
        lattice_params = f"{phase['a']};{phase['b']};{phase['c']}"
        angles = f"{phase['alpha']};{phase['beta']};{phase['gamma']}"
        if phase["SpaceGroup"] == None:
            header += f"{lattice_params}\t{angles}\t{phase['Name']}\t{phase['Symmetry']}\n"
        else:
            header += f"{lattice_params}\t{angles}\t{phase['Name']}\t{phase['Symmetry']}\t{phase['SpaceGroup']}\n"
    # Write the CTF file
    with open(output_path, 'w') as ctf_file:
        # Write header
        ctf_file.write(header)
        
        # Write column headers
        ctf_file.write("Phase X Y Bands Error Euler1 Euler2 Euler3 MAD BC BS\n")
        
        # Write data rows
        for row in data:
            phase = row[1]
            if phase == "Zero":
                offset = 1
                phase = "-".join(row[1:3])
                #phase = -1
                phase = phase_dict[phase]
            else:
                phase = phase_dict[phase]
                offset = 0
            x = "{:.6f}".format(float(row[2+offset]))
            y = "{:.6f}".format(float(row[3+offset]))
            phi1 = "{:.6f}".format(float(row[4+offset]))
            PHI = "{:.6f}".format(float(row[5+offset]))
            phi3 = "{:.6f}".format(float(row[6+offset]))
            mad = "{:.6f}".format(float(row[7+offset]))
            bc = row[9+offset]
            bs = row[10+offset]
            line = f"{phase} {x} {y} 0 0 {phi1} {PHI} {phi3} {mad} {bc} {bs}\n"
            ctf_file.write(line)

if __name__ == "__main__":
    path_to_txt = "./txt_files/all"
    path_to_ctf = "./ctf_files"
    filenames = glob.glob(path_to_txt+"/*.txt")
    for filename in filenames:
        filename = filename.split("/")[-1].split(".")[0]
        input_txt_file = f'{path_to_txt}/{filename}.txt'
        output_ctf_file = f'{path_to_ctf}/{filename}.ctf'
        header_info = {
            'XCells': 450,  # Adjust according to your data
            'YCells': 450,  # Adjust according to your data
            'XStep': 0.1,   # Adjust according to your data
            'YStep': 0.1,   # Adjust according to your data
            'Phases': [
                {
                    'Index': 1,
                    'Name': 'HCP',
                    'Formula': 'Ti64',
                    'Symmetry': 9,  # Symmetry number for hexagonal structures
                    'a': 3.209,
                    'b': 3.209,
                    'c': 5.210,
                    'alpha': 90,
                    'beta': 90,
                    'gamma': 120,
                    'SpaceGroup':1,
                },
                {
                    'Index': 2,
                    'Name': 'BCC',
                    'Formula': 'Ti64',
                    'Symmetry': 11,  # Symmetry number for cubic structures
                    'a': 2.870,
                    'b': 2.870,
                    'c': 2.870,
                    'alpha': 90,
                    'beta': 90,
                    'gamma': 90,
                    'SpaceGroup':229,
                }
            ]
        }
        phase_dict = {"Zero-solution":0,
                      "Titanium-Hexagonal":1,
                      "Titanium-Cubic":2}
        # Read the TXT file
        data = read_txt_file(input_txt_file)
        # Write the CTF file
        write_ctf_file(data, output_ctf_file, header_info, phase_dict)

