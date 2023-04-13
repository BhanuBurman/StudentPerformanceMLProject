from setuptools import find_packages,setup
from typing import List
HYPEN_E = "-e ."


def get_requirments(file_path:str) -> List[str]:
        #This function returns a list of requirements
        requirements = []
        with open(file_path) as f:
            requirements = f.readlines()
            requirements = [req.replace("\n","")for req in requirements]
            if HYPEN_E in requirements:
                requirements.remove(HYPEN_E)
        return requirements
setup(
    name="StudentPerformance",
    version = "0.0.1",
    author = "Bhanu",
    author_email = "bhanuburman@gmail.com",
    packages = find_packages(),
    #"Here a list is needed to return but this we will get from the .txt file"
    install_requires = get_requirments("requirements.txt")
)