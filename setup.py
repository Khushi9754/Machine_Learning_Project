from setuptools import find_packages,setup
from typing import List

hyphen_e = "-e ."

def get_requirements(file_path:str)->List[str]:
    #"""install all packages of requirement.txt file"""
    
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [i.replace("/n"," ") for i in requirements]

        if hyphen_e in requirements:
            requirements.remove(hyphen_e)
            
    return requirements    
        
setup(
    
    name = 'ML_PROJECT',
    version = '0.0.1',
    author = 'Khushi',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
    
)