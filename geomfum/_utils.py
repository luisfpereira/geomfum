import importlib


def has_package(package_name):
    """Check if package is installed.

    Parameters
    ----------
    package_name : str
        Package name.
    """
    return importlib.util.find_spec(package_name) is not None
