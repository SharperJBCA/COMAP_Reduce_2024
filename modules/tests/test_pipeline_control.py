# test_pipeline_control
#
# Description: 
#   Want to test how the pipeline control functions handle:
#     1) Running modules within the Master pipeline
#     2) How it executes pipelines within sub modules. 
#     3) How it handles arguments to pipeline modules. 
import sys 
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from pipeline_control.Pipeline import run_pipeline


def test_pipeline_master_execution():
    param_filename = Path(__file__).parent / "test_pipeline_control.toml"
    modules = run_pipeline(param_filename)
    assert modules[0].parameter_file == 'modules/tests/test_pipeline_control_mock_module.toml'
 

if __name__ == "__main__":
    test_pipeline_master_execution()
