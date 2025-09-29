from modules.pipeline_control.Pipeline import run_pipeline
import sys
import matplotlib
matplotlib.use('agg')

if __name__ == "__main__":

    run_pipeline(sys.argv[1])