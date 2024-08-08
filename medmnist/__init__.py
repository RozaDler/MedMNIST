from medmnist.info import __version__, HOMEPAGE, INFO
try:
    from medmnist.dataset import (PathMNIST, ChestMNIST, DermaMNIST, OCTMNIST, PneumoniaMNIST, RetinaMNIST,
                                  BreastMNIST, BloodMNIST, TissueMNIST, OrganAMNIST, OrganCMNIST, OrganSMNIST,
                                  OrganMNIST3D, NoduleMNIST3D, AdrenalMNIST3D, FractureMNIST3D, VesselMNIST3D, SynapseMNIST3D)
    from medmnist.evaluator import Evaluator
except:
    print("Please install the required packages first. " +
          "Use `pip install -r requirements.txt`.")


"""
This file is used to initialize the medmnist package and make it easier to import the required classes and functions.
it imports version info, homepage url, and dataset info from info.py 
the try block imports dataset classes from dataset.py and 'Evaluator' class from evaluator.py 
You dont need to interact with this file directly. It's used to make imports easier when you use the package.
"""