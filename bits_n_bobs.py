import gnuplotlib as gp
import numpy as np
from pyfiglet import Figlet


def plot_vector(vector):
    x = np.array(list(range(len(vector))))
    gp.plot( x, vector, _with='lines', terminal='dumb 180,30')

def bigtext(string):
    f = Figlet(font='big', width=130)
    return f.renderText(string)

if __name__ == '__main__':
    plot_vector(np.array([1,2,3,4,5,6,7,8,9,10]))
