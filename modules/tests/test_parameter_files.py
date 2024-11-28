import pytest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from parameter_files.parameter_files import read_parameter_file, parse_function_string

def test_parse_function_string():
    package, module, args = parse_function_string("control.Package.SubPackage.Module(arg1='val1',arg2=10)")
    assert package == "control.Package.SubPackage"
    assert module == "Module"
    assert args == {"arg1": "val1", "arg2": 10}

def test_read_parameter_file():
    file_path = Path(__file__).parent / "test_parameter_file.toml"
    parameters = read_parameter_file(file_path)
    test_parameters = {"Master": {"sql_database":"sqlite:///none_database.db",
                                    "pipeline": ['Package.SubPackage.Module1(arg1="var1", arg2=10)',
                                                  'Package.SubPackage.Module2(arg1="var2", arg2=20)'],
                                    "_pipeline": [{"package": "Package.SubPackage", "module": "Module1", "args": {"arg1": "var1", "arg2": 10, "arg3":"hello"}},
                                                  {"package": "Package.SubPackage", "module": "Module2", "args": {"arg1": "var2", "arg2": 20}}]},
                        "Package":{"SubPackage":{"Module1": {"arg3":'hello'}}}
                        }  
    assert parameters == test_parameters

    