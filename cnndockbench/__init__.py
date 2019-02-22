import inspect
import os



def home(data=None):
    import cnndockbench

    homeDir = inspect.getfile(cnndockbench)
    homeDir = os.path.dirname(homeDir)

    if data is None:
        return homeDir

    return os.path.join(homeDir, data)

