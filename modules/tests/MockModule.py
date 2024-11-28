# MockModule.py
#
# Description:
#  A dummy module for testing the pipeline 
import logging 

class MockModule:

    def __init__(self, **kwargs):
        logging.info("Initializing MockModule")
        for k,v in kwargs.items():
            logging.info(f'{k}: {v}')
            setattr(self, k, v) 
            
    def run(self):
        logging.info("Running MockModule")
        pass

    def __str__(self):
        return "MockModule"
    
    def __repr__(self):
        return "MockModule"
    
