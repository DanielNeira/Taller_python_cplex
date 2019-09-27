import cplex
import sys
from pylab import *

class MyCallback(cplex.callbacks.MIPInfoCallback):
    
    def __call__(self):
        if self.has_incumbent():
            self.incobjval.append(
                self.get_incumbent_objective_value())
            self.bestobjval.append(
                self.get_best_objective_value())
def main():
    cpx = cplex.Cplex()
    cb = cpx.register_callback(MyCallback)
    cb.incobjval, cb.bestobjval = [], []
    cpx.read('Tarea.lp')
    cpx.solve()
    # plot obj value
    size = len(cb.incobjval)
    plot(range(size),cb.incobjval)
    plot(range(size),cb.bestobjval)
    #save to PNG file
    savefig('cpxplot.png')

if __name__ == '__main__':
    main()