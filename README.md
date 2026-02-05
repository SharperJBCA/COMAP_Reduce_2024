# COMAP Data Processing Pipeline

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation & Environment Setup](#installation--environment-setup)
- [Input Data & Databases](#input-data--databases)
- [Configuration](#configuration)
    - [Level2 TOD Processing](#level-2-tod-processing)
    - [Level2 Map-Making](#level-2-map-making)
    - [RRL Map-Making](#rrl-map-making)
- [Running the Pipeline](#running-the-pipeline)
- [Extending the Pipeline](#extending-the-pipeline)

## Overview
The COMAP pipeline processes raw COMAP data into calibrated timelines and final calibrated science-ready maps.  
It manages:
- Reading raw level 1 data.  
- Processing and calibration into level 2 data.  
- Calibration source fitting.
- Destriping Map-making.  

The pipeline is modular and configurable through TOML configuration files.

## Repository Structure
- **run.py** — Main entry point to run the pipeline.  
- **modules/** — Processing modules:  
  - `DataAcquisition` — Download raw data from Presto (via `rsync`).  
  - `ProcessLevel2` — Low-level processing of Level 1 data.  
  - `MapMaking` — Destriping map-making routines.  
  - `SQLModule` — Database schema and SQLAlchemy classes.  
  - `utils` — Utility functions (coordinates, source fluxes, etc.).  
  - `scripts` — Tools for querying the databases.  
- **parameter_files/** — TOML configuration files for pipeline runs.  
- **databases/** — SQLite databases (file tracking, noise stats, calibration factors).  
- **logs/** — Log outputs from pipeline runs.  

## Installation & Environment Setup
### Prerequisites
- **Python:** 3.11 
- **Conda:** [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda installed  

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).  
### Setup Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/sharperJBCA/COMAP_Reduce_2024
   cd COMAP_Reduce_2024

2. Create copy of comap_reduce conda environment
    ```bash
    conda env create -f environment.yml
    conda activate comap_reduce
    ``` 

3. Compile slalib routines
    ```bash 
    cd utils 
    make
    cd ..
    ``` 

4. Compile utils routines
    ```bash
    cd utils
    python setup_bin_funcs.py build_ext --inplace 
    python setup_median_filter.py build_ext --inplace
    python setup_mean_filter.py build_ext --inplace
    ``` 

## Input Data & Databases

### Database
The pipeline requires the COMAP_manchester.db SQLite database.
 - Copy the existing version from the Manchester system:
    ```bash
    cp /scratch/nas_cbassarc/sharper/work/COMAP/COMAP_Reduce_2024/databases/COMAP_manchester.db
    ```    

### Locations of data 
Data is stored on networked NAS drives. These should be accessible from all of the JBCA machines
but may need to be mounted on your workstation. 

Level 1 data: 
 - `nas_comap1/sharper/COMAP/data/`
 - `nas_comap2/sharper/COMAP/data/`
 - `nas_core/sharper/COMAP/data/`
 - `nas_core2/sharper/COMAP/data/`

Level 2 data: 
 - `nas_core2/sharper/COMAP/level2/`


## Configuration 

The primary configuration is in `parameter_files/main_run.toml`.
This defines which modules run in the pipeline and their parameters.

### Level 2 TOD Processing 
call: `modules.ProcessLevel2.ProcessLevel2.Level2Pipeline()`

Level 2 processing configuration is defined by `parameter_files/level2_processing.toml`

#### Level 2 processing configuration information
todo

### Level 2 Map-Making
call: `modules.MapMaking.MapMaking.MapMaking()`

Level 2 map-making configuration is defined by `parameter_files/level2_mapmaking.toml`

#### Level 2 map-making configuration information
todo

### RRL Map-Making
call: `modules.MapMaking.MapMaking.LineMapMaking()`

RRL map-making configuration is defined by `parameter_files/level2_line_mapmaking.toml`

#### RRL map-making configuration information

The `pipeline` keyword expects a list of calls to the `CreateLineMap` class that can be used to define the creation of
maps for different line frequencies and regions. The keywords for the `CreateLineMap` class are:

- `line_frequency` - Central frequency of the RRL in GHz 
- `wcs_def` - Predefined coordinate system found in map_making_configs/wcs_definitions.toml. For galactic data use "Galactic".
- `source` - This is the field name to map. Use this to map just small sections of the sky. See COMAP wiki for details of field names.
- `source_group` - Can be Galactic or Foreground. If source is not defined then all observations within source_group will be mapped.
- `map_name` - Prefix for the output map file.
- `database_file` - The COMAP database needed for noise statistics and calibration factors. 
- `output_dir` - output directory of the maps 
- `wcs_def_file` - leave as: map_making_configs/wcs_definitions.toml 
- `feeds` - List of feeds to use between 1 and 19. 

Key parameters you may want to change are:
- `line_frequency`
- `source`
- `feeds` 

Under the `[modules.MapMaking.MapMakingModules.CreateLineMap.CreateLineMap]` header keywords can be set globally for all runs.

## Running the Pipeline 

Once the configuration files are setup the pipeline can be run with simply:
    ```bash
    python run.py parameter_files/main_run.toml
    ``` 

## Extending the Pipeline

New functionality can be added to the pipeline by writing additional **modules**.  
Modules are typically placed inside `modules/` (e.g., `modules/MyModule/`). Each module should define a class that follows the pipeline conventions:

### 1. Module Structure
- **Base class**: Inherit from `BaseCOMAPModule` when possible (provides logging, HDF5 retry helpers, and standard behaviour).  
- **Main entry point**: Implement a `run(self, file_info: COMAPData)` method, which will be called for each observation.  
- **Helper methods**:  
  - `already_processed(file_info)` → avoid reprocessing existing results.  
  - `fit_*` → core data analysis step(s).  
  - `save_*` → write results into the Level 2 HDF5 files.  
  - `plot_*` (optional) → produce diagnostic plots.  

### 2. Example:

Below is a template class that should fit into the COMAP ProcessLevel2 modules:

```python
from modules.pipeline_control.Pipeline import BaseCOMAPModule
from modules.SQLModule.SQLModule import COMAPData

class MyNewModule(BaseCOMAPModule):
    def __init__(self, param1=True, output_dir="outputs/MyNewModule", overwrite=False):
        self.param1 = param1
        self.output_dir = output_dir
        self.overwrite = overwrite

    def already_processed(self, file_info: COMAPData, overwrite: bool = False) -> bool:
        """ Check if results already exist in the Level 2 file """
        if overwrite:
            return False 
            
        with RetryH5PY(file_info.level2_path, 'r') as lvl2:
            return "level2/my_new_results" in lvl2

    def run(self, file_info: COMAPData) -> None:
        """ Main entry point called by the pipeline """
        if self.already_processed(file_info):
            return
        # --- analysis logic here ---
        results = self.do_analysis(file_info)
        self.save_results(file_info, results)

    def do_analysis(self, file_info: COMAPData):
        """ Example processing routine """
        # Read level 1 data, perform computations, return numpy arrays
        return {"example": 42}

    def save_results(self, file_info: COMAPData, results: dict):
        """ Save results into the Level 2 HDF5 file """
        with RetryH5PY(file_info.level2_path, 'a') as lvl2:
            grp = lvl2.require_group("level2/my_new_results")
            for key, value in results.items():
                if key in grp:
                    del grp[key]
                grp.create_dataset(key, data=value)
```

### 3. Register Module 

To use the new module in a pipeline run new modules should have their own directory and `__init__.py`. For example if it is a ProcessLevel2 module it should be structured as: 

- `modules/ProcessLevel2/MyNewModule/__init__.py`
- `modules/ProcessLevel2/MyNewModule/MyNewModule.py`

Second, add the class to the configuration file:

```toml
pipeline = [
  "modules.MyModule.MyNewModule.MyNewModule(param1=True, output_dir='outputs/new_results')"
]
```