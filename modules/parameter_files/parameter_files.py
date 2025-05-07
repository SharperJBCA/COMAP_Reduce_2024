import toml 
from functools import reduce
import operator

def get_from_dict(data_dict, map_list, module):
    d = reduce(operator.getitem, map_list, data_dict)
    if module in d:
        return d[module]
    else:
        return {}

def parse_function_string(function_str):
    """
    Parse a string of format Package.ModuleDip(arg1=val1,arg2=val2)
    Returns package name, module name, and arguments dictionary
    
    Args:
        function_str (str): The function string to parse
        
    Returns:
        tuple: (package_name, module_name, arguments_dict)
    """
    # Split on parentheses to separate function name from arguments
    function_parts = function_str.split('(')
    if len(function_parts) != 2 or not function_parts[1].endswith(')'):
        raise ValueError("Invalid function string format")
        
    # Get the full name (Package.Module)
    full_name = function_parts[0]
    
    # Split full name into package and module
    name_parts = full_name.rsplit('.',1)
    if len(name_parts) != 2:
        raise ValueError("Invalid package.module format")
        
    package = name_parts[0]
    module = name_parts[1]
        
    # Parse arguments
    args_str = function_parts[1].rstrip(')')
    args_dict = {}
    
    if args_str:
        # Split arguments on commas and process each key-value pair
        arg_pairs = args_str.split(', ')
        for pair in arg_pairs:
            print(pair)
            if '=' not in pair:
                raise ValueError("Invalid argument format")
            key, value = pair.split('=')
            if value[0] == '[':
                value = value[1:-1].split(',')
                value = [v.replace('"','').strip() for v in value]
            elif value.startswith('"') and value.endswith('"') or value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            else:
                try:
                    # Try to convert to a number
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                finally:
                    value = value

            args_dict[key.strip()] = value
    
    return package, module, args_dict

def read_parameter_file(file_path, primary_key='Master'):
    """Read a parameter file and return the parameters as a dictionary."""

    parameters = toml.load(file_path) 

    parameters[primary_key]['_pipeline'] = [] 
    for step in parameters[primary_key]["pipeline"]:
        # Read the "pipeline" string and parse it
        package, module, args = parse_function_string(step)

        # Also see if there are arguments defined in toml file

        package_list = package.split('.')
        args = {**args, **get_from_dict(parameters, package_list, module)}

        parameters[primary_key]['_pipeline'].append({'package': package, 'module': module, 'args': args})

    return parameters