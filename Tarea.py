#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  26 01:34:09 2018

@author: Daniel Neira
"""
import time
import pandas as pd
import numpy as np
import cplex
from cplex import Cplex
from cplex.exceptions import CplexError
import sys


def crear_dat():
	ruta = 'data/tsiligirides_problem_1_budget_85.txt'
	archivo = pd.read_csv(ruta, header=None, delim_whitespace=True)
	tiempo_max=archivo.iloc[-1].values
	tiempo_max=int(tiempo_max[0])
	dist=[]
	for i in range(31):
	    a=archivo.iloc[i]
	    a=np.array(a)
	    a=np.delete(a,2)
	    for j in range(31):
	        if i!=j:
	            b=archivo.iloc[j]
	            b=np.array(b)
	            b=np.delete(b,2)
	            dist.append(int(np.linalg.norm(a-b)))
	        elif i==j:
	            dist.append(99999)
	distancias=np.array(dist)
	distancias=distancias.reshape(31,31)
	#print(distancias)
	beneficios=[]
	for i in range(31):
	    c= archivo.iloc[i]
	    c = [int(i) for i in c]
	    del c[0]
	    del c[0]
	    c=c[0]
	    beneficios.append(c)
	#print(beneficios)
	nodos = len(beneficios)
	return distancias, beneficios, tiempo_max, nodos

def resolver_modelo(t, S, t_max, N, T_exec, Memory, M=99999):
	Model = cplex.Cplex()

	#Variable de decision----------------------------------------------------------------------------------------------

	x_vars = np.array([["x(" + str(i) + "," +str(j)+ ")"  for j in range(N)] for i in range(N)])
	x_varnames = x_vars.flatten()
	x_vartypes = 'B'*len(x_varnames)
	x_varlb = [0.0]*len(x_varnames)
	x_varub = [1.0]*len(x_varnames)
	S_mat = np.empty([len(S),len(S)], dtype=float)
	for i in range(len(S)):
		for j in range(len(S)):
			if i == j:
				S_mat[i,j] = 0.0
			else:
				S_mat[i,j] = float(S[i])
	#print(S_mat)
	x_varobj =  S_mat.flatten()
	print(x_varnames)
	print('tamaños:',len(x_varnames),len(x_varobj),'N:',N)
	Model.variables.add(obj = x_varobj, lb = x_varlb, ub = x_varub, types = x_vartypes, names = x_varnames)

	u_vars = np.array(['u('+str(i)+')' for i in range(N)])
	u_varnames = u_vars.flatten()
	u_vartypes = 'C'*len(u_varnames)
	u_varlb = [0.0]*len(u_varnames)
	u_varub = [cplex.infinity]*len(u_varnames)
	u_varobj = []#[0.0]*len(u_varnames)

	Model.variables.add(obj = u_varobj, lb = u_varlb, ub = u_varub, types = u_vartypes, names = u_varnames)
	#Restriccions -----------------------------------------------------------------------------------------------------

	#Restriccion 1.1
	row1 = []
	val1 = []
	for j in range(1,N):#equivalente a (2,N)
		row1.append(x_vars[0,j])
		val1.append(1.0)
	Model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = row1, val = val1)], senses = 'E', rhs = [1.0])

	#Restriccion 1.2
	row2 = []
	val2 = []	
	for i in range(N-1):
		row2.append(x_vars[i,N-1])
		val2.append(1.0)
	Model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = row2, val= val2)], senses = 'E', rhs = [1.0])

	#Restriccion 2.1
	for k in range(1,N-1):
		row3 = []
		val3 = []
		for i in range(N-1):
			if k != i:
				row3.append(x_vars[i,k])
				val3.append(1.0)
		for j in range(1,N):
			if k != j:
				row3.append(x_vars[k,j])
				val3.append(-1.0)
		Model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = row3, val= val3)], senses = 'E', rhs = [0.0])

	#Restriccion 2.2
	for k in range(1,N-1):
		row4 = []
		val4 = []
		for j in range(1,N):
			row3.append(x_vars[k,j])
			val3.append(1.0)		
		Model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = row4, val= val4)], senses = 'L', rhs = [1.0])			
	
	#Restriccion 3
	row5 = []
	val5 = []
	for i in range(N-1):
		for j in range(1,N):
			if(i!=j):
				row5.append(x_vars[i,j])
				val5.append(float(t[i,j]))
	Model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = row5, val= val5)], senses = 'L', rhs = [float(t_max)])

	#Restriccion 4.1
	for i in range(1,N):
		row6 = []
		val6 = []
		row6.append(u_vars[i])
		val6.append(1.0)
		Model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = row6, val= val6)], senses = 'G', rhs = [1.0])
		Model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = row6, val= val6)], senses = 'L', rhs = [float(N)])

	#Restriccion 5
	for i in range(1,N):
		for j in range(1,N):
			if i!=j:
				row7 = []
				val7 = []
				row7.append(u_vars[i])
				val7.append(1.0)
				row7.append(u_vars[j])
				val7.append(-1.0)						
				row7.append(x_vars[i,j])
				val7.append(float(N-1))		
				Model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = row7, val= val7)], senses = 'L', rhs = [float(N-2)])
	
	#Restriccion extra 1
	row1 = []
	val1 = []
	for i in range(N):
		row1.append(x_vars[i,0])
		val1.append(1.0)
	Model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = row1, val = val1)], senses = 'E', rhs = [0.0])
	#Restriccion extra 2
	row1 = []
	val1 = []
	for j in range(N):
		row1.append(x_vars[N-1,j])
		val1.append(1.0)
	Model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = row1, val = val1)], senses = 'E', rhs = [0.0])
	#Setting Solver Parameters---------------------------------------------------------------------------------------------
	# Sentido de Optimización
	Model.objective.set_sense(Model.objective.sense.maximize)
	# Setting Tiempo limite optimización global
	Model.parameters.timelimit.set(float(T_exec))
	# Setting Memoria Ram en computador
	Model.parameters.workmem.set(Memory)
	#####guardar Lp
	Model.write("Tarea.lp")
	#Solving model---------------------------------------------------------------------------------------------------------
	tiemp_antes = Model.get_time()
	Model.solve()
	tiemp_despues = Model.get_time()
	tiemp_ejecucion=tiemp_despues - tiemp_antes
	def show_solution():
		x_sol =[]
		x_path=[]
		for i in range(N):
			for j in range(N):
				if(Model.solution.get_values("x("+str(i)+","+str(j)+")")!=0.0):
					x_sol.append("x("+str(i)+","+str(j)+")")
					x_path.append((i,j))
		u_sol = []
		for i in range(1,N):
			if (Model.solution.get_values('u('+str(i)+')') != 0.0):
				u_sol.append('u('+str(i)+')')
		return x_sol, x_path, u_sol

	def show_solution_1():
		var_sol = list()
		for i in range(Model.variables.get_num()):
			var_sol.append({Model.variables.get_names(i): Model.solution.get_values(i)})
		dis_cero = list()
		for i in range(len(var_sol)):
			for key in var_sol[i]:
				if var_sol[i][key] != 0.0:
					dis_cero.append({key:var_sol[i][key]})
		print(dis_cero)
	def show_solution_gap():
		var_sol = list()
		for i in range(Model.variables.get_num()):
			var_sol.append({Model.variables.get_names(i): Model.solution.pool.get_values(0,i)})
		dis_cero = list()
		for i in range(len(var_sol)):
			for key in var_sol[i]:
				if var_sol[i][key] != 0.0:
					dis_cero.append({key:var_sol[i][key]})
		print(dis_cero)
		return dis_cero

	if Model.solution.get_status() == 101: #MIP_optimal
		estado = 'OPTIMO'
		obj_function = Model.solution.get_objective_value()
		x_sol, x_path, u_sol = show_solution()
		show_solution_1()
	elif Model.solution.get_status() == 107 or Model.solution.get_status() == 102 or Model.solution.get_status() == 131 or Model.solution.get_status() == 113:
	#MIP_time_limit_feasible, optimal_tolerance, MIP_dettime_limit_feasible, MIP_abort_feasible
		estado = Model.solution.status[Model.solution.get_status()]
		gap = Model.solution.MIP.get_mip_relative_gap()
		print('Aqui el gaaaap:',gap)
		obj_function = Model.solution.get_objective_value()
		dis_cero =show_solution_gap()
		#= Model.solution.pool.get_values(0)
		ub = Model.solution.MIP.get_best_objective()
		print('Funcion obj:',obj_function)
		print('Lower bound:',ub)
		x_sol,x_path,u_sol = [0,0,0]
	elif Model.solution.get_status() == 103 or Model.solution.get_status() == 132 or Model.solution.get_status() == 114 or Model.solution.get_status() == 115:
	#MIP_infeasible, MIP_dettime_limit_infeasible, MIP_abort_infeasible, MIP_optimal_infeasible
		estado = Model.solution.status[Model.solution.get_status()]
		obj_function = 0
		x_sol,x_path,u_sol = [0,0,0]
		MIP_time_limit_infeasible = 108
	else:
		estado = Model.solution.status[Model.solution.get_status()]
		obj_function = 0
		x_sol,x_path,u_sol = [0,0,0]

	return estado, obj_function, x_sol, x_path, tiemp_ejecucion			

def escribrir(instancia, estado, obj_function, x_sol, tiemp_ejecucion):
	"""
	Esta función toma los strings motor, solucion, x_sol, tiemp_ejecutado, y si plot es True grafica la sol
	primera columna: instancia, segunda: función objetivo y tercera: tiemp_ejecucion
	"""	
	archivo = open('resultados.txt','a')
	archivo.write(str(instancia)+';'+str(estado)+';'+str(obj_function)+';'+str(tiemp_ejecucion)+';\n')
	archivo.close()

def correr():
	distancias, beneficios, tiempo_max, nodos = crear_dat()
	estado, obj_function, x_sol, x_path, tiemp_ejecucion = resolver_modelo(distancias, beneficios, tiempo_max, nodos, T_exec, Memory)
	print('Estado:',estado,'Tiempo Ejecución:', tiemp_ejecucion)
	if estado=='OPTIMO':
		print('Función Objetivo:', obj_function)
		print('SOOOOOOOOOOL:',x_sol)
	#escribrir(instancia, estado, obj_function, x_sol, tiemp_ejecucion)


############CORRER#################
T_exec = 10800 #Tiempo de ejecución máximo = a 3 horas por ahora
Memory = 4096 #Memoria RAM que le entregamos a Cplex para correr modelos, set a 4 GB
correr()
