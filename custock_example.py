import numpy as np
import cplex
from cplex import Cplex
from cplex.exceptions import CplexError


def resolver_modelo(L,l,d,B,M=99999):
	Model = cplex.Cplex()
	S = len(l)
	#Variable de decision----------------------------------------------------------------------------------------------
	x_vars = np.array([["x(" + str(i) + "," +str(j)+ ")"  for j in range(B)] for i in range(S)])
	x_varnames = x_vars.flatten()
	x_vartypes = 'I'*len(x_varnames)
	x_varlb = [0.0]*len(x_varnames)
	x_varub = [cplex.infinity]*len(x_varnames)
	x_varobj =[]
	Model.variables.add(obj = x_varobj, lb = x_varlb, ub = x_varub, types = x_vartypes, names = x_varnames)

	y_vars = np.array(['y('+str(i)+')' for i in range(B)])
	y_varnames = y_vars.flatten()
	y_vartypes = 'B'*len(y_varnames)
	y_varlb = [0.0]*len(y_varnames)
	y_varub = [1.0]*len(y_varnames)
	y_varobj = [1.0]*len(y_varnames)

	Model.variables.add(obj = y_varobj, lb = y_varlb, ub = y_varub, types = y_vartypes, names = y_varnames)
	#Restriccions -----------------------------------------------------------------------------------------------------

	#Restriccion 1
	for b in range(B):
		for s in range(S):
			row1 = []
			val1 = []
			row1.append(y_vars[b])
			val1.append(float(M))
			row1.append(x_vars[s,b])
			val1.append(-1.0)
			Model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = row1, val = val1)], senses = 'G', rhs = [0.0],
				names=['c1-'+str(b)+'-'+str(s)])

	#Restriccion 2
	for b in range(B):
		row2 = []
		val2 = []
		for s in range(S):
			row2.append(x_vars[s,b])
			val2.append(float(l[s]))
		Model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = row2, val = val2)], senses = 'L', rhs = [float(L)],
			names=['c2-'+str(b)])
	
	#Restriccion 3
	for s in range(S):
		row3 = []
		val3 = []
		for b in range(B):
			row3.append(x_vars[s,b])
			val3.append(1.0)
		Model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = row3, val = val3)], senses = 'G', rhs = [float(d[s])],
			names=['c3-'+str(s)])
			

	#Setting Solver Parameters---------------------------------------------------------------------------------------------
	# Sentido de Optimización
	Model.objective.set_sense(Model.objective.sense.minimize)
	#resolver
	Model.solve()


L = 115 #largo
l = [25, 40, 50, 55, 70] #corte
d = [50, 136, 24, 8, 30] #demanda de cada corte
B = 1000
resolver_modelo(L,l,d,B)













