import numpy as np 
import logging 

class AMatrixGainFilter: 

    def __init__(self, templates : np.ndarray) -> None:
        self.templates = templates
        self.M = self.templates.T.dot(self.templates)

    def create_b(self, d : np.ndarray) -> np.ndarray:
        """
        Create the b vector for the gain solution

        Arguments:
        ----------
        d - length frequency*time
        """

        return self.templates.T.dot(d) 

    def __call__(self,g):
        A = self.M.dot(g)  
        return A 