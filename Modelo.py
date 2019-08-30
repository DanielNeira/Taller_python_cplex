#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 01:34:09 2018

@author: Daniel Neira
"""
import time
import numpy as np
import cplex
from cplex import Cplex
from cplex.exceptions import CplexError
import sys
import networkx as nx
import matplotlib.pyplot as plt

def crear_dat(NUM, name, tmax, rutas):
	#abriendo datos
	file = open('data/'+name+'.txt')
	data = file.readlines()
	file.close()  
	Q = data[4].split()[1] 
	#capacidad de vehiculos

	#EL primer nodo es el deposito!!
	lista =[]##contiene info de text
	for i in range(9, NUM+10):
		lista.append({})
		index = len(lista) - 1
		lista[index]['x'] = float(data[i].split()[1])
		lista[index]['y'] = float(data[i].split()[2])
		lista[index]['d'] = int(data[i].split()[3])
		lista[index]['tw_ini'] = int(data[i].split()[4])
		lista[index]['tw_fin'] = int(data[i].split()[5])
		lista[index]['st'] = int(data[i].split()[6])

	###########################lista para propio
	listapropio = []
	for i in range(rutas-1):
		listapropio.append(lista[0])
	listapropio = listapropio + lista
	c2 = np.ones((len(listapropio),len(listapropio)))*(0)
	for i in range(len(listapropio)):
		for j in range(len(listapropio)):
			if i != j:
				#puedes truncarlo con np.trunc
				c2[i][j] = np.sqrt(pow(listapropio[j]['x']-listapropio[i]['x'],2) + pow(listapropio[j]['y']-listapropio[i]['y'],2))
	c2 = c2.astype(int) ##truncamos
	t2 = c2
	index = len(listapropio)
	o2 = np.zeros(index)
	e2 = np.zeros(index)
	d2 = np.zeros(index)
	s2 = np.zeros(index)
	x = np.zeros(index)
	y = np.zeros(index)
	for i in range(index):
		o2[i] = int(listapropio[i]['tw_ini'])
		e2[i] = int(listapropio[i]['tw_fin'])
		d2[i] = int(listapropio[i]['d'])
		s2[i] = int(listapropio[i]['st'])
		x[i] = float(listapropio[i]['x'])
		y[i] = float(listapropio[i]['y'])
	return Q, c2, t2, o2, e2, d2, s2, x, y

def resolver_modelo(N,K,Q,c,o,e,dem,s,t,tmax, T_exec, Memory,M=99999):
	Model = cplex.Cplex()
	before = time.clock()

	#Variable de decision----------------------------------------------------------------------------------------------

	x_vars = np.array([["x(" + str(i) + "," +str(j)+ ")"  for j in range(K+N)] for i in range(K+N)])
	x_varnames = x_vars.flatten()
	x_vartypes = 'B'*len(x_varnames)
	x_varlb = [0.0]*len(x_varnames)
	x_varub = [1.0]*len(x_varnames)
	x_varobj = c.flatten() ##lo pongo como copia porque no pude hacerlo de otra forma
	Model.variables.add(obj = x_varobj.astype(float), lb = x_varlb, ub = x_varub, types = x_vartypes, names = x_varnames)

	y_vars = np.array([['y('+str(i)+','+str(j)+')' for j in range(K+N)] for i in range(K+N)])
	y_varnames = y_vars.flatten()
	y_vartypes = 'C'*len(y_varnames)
	y_varlb = [0.0]*len(y_varnames)
	y_varub = [cplex.infinity]*len(y_varnames)
	y_varobj = []#[0.0]*len(y_varnames)

	Model.variables.add(obj = y_varobj, lb = y_varlb, ub = y_varub, types = y_vartypes, names = y_varnames)	

	u_vars = np.array(['u('+str(i)+')' for i in range(1, N+K)])
	u_varnames = u_vars.flatten()
	u_vartypes = 'C'*len(u_varnames)
	u_varlb = [0.0]*len(u_varnames)
	u_varub = [cplex.infinity]*len(u_varnames)
	u_varobj = []#[0.0]*len(u_varnames)

	Model.variables.add(obj = u_varobj, lb = u_varlb, ub = u_varub, types = u_vartypes, names = u_varnames)

	a_vars = np.array(['a('+str(i)+')' for i in range(K, N+K)])
	a_varnames = a_vars.flatten()
	a_vartypes = 'C'*len(a_varnames)
	a_varlb = [0.0]*len(a_varnames)
	a_varub = [cplex.infinity]*len(a_varnames)
	a_varobj = []#[0.0]*len(a_varnames)

	Model.variables.add(obj = a_varobj, lb = a_varlb, ub = a_varub, types = a_vartypes, names = a_varnames)
				
	#Restriccions -----------------------------------------------------------------------------------------------------

	#Restriccion 1
	row1 = []
	val1 = []
	for j in range(1,N+K):
		row1.append(x_vars[0,j])
		val1.append(1.0)
	Model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = row1, val = val1)], senses = 'E', rhs = [1.0])

	#Restriccion 2
	row2 = []
	val2 = []	
	for j in range(1,N+K):
		row2.append(x_vars[j,0])
		val2.append(1.0)
	Model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = row2, val= val2)], senses = 'E', rhs = [1.0])

	#Restriccion 3
	for d in range(K):
		row3 = []
		val3 = []
		for j in range(K):
			#if d!=j:
			#	row3.append(x_vars[j,d])
			#	val3.append(1.0)
			row3.append(x_vars[j,d])
			val3.append(1.0)
		Model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = row3, val= val3)], senses = 'E', rhs = [0.0])

	#Restriccion 4
	for i in range(K,N+K):
		row4 = []
		val4 = []
		for j in range(N+K):
			if i!=j:
				row4.append(x_vars[i,j])
				val4.append(1.0)		
		Model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = row4, val= val4)], senses = 'E', rhs = [1.0])			
	
	#Restriccion 5	
	for d in range(1,K):
		row5 = []
		val5 = []
		for j in range(N+K):
			if(j!=d):
				row5.append(x_vars[d,j])
				val5.append(1.0)
		Model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = row5, val= val5)], senses = 'L', rhs = [1.0])

	#Restriccion 6
	for i in range(1,N+K):
		row6=[]
		val6=[]
		for j in range(N+K):
			if i!=j:
				row6.append(x_vars[j,i])
				val6.append(1.0)
		for j in range(N+K):
			if i!=j:
				row6.append(x_vars[i,j])
				val6.append(-1.0)
		Model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = row6, val= val6)], senses = 'E', rhs = [0.0])

	#Restriccion 7
	for i in range(1,N+K):
		for j in range(1,N+K):
			row7 = []
			val7 = []
			if i!=j:
				row7.append(u_vars[i-1])
				val7.append(1.0)
				row7.append(u_vars[j-1])
				val7.append(-1.0)						
				row7.append(x_vars[i,j])
				val7.append(float(M))		
				Model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = row7, val= val7)], senses = 'L', rhs = [float(M-t[i][j]-s[i])])
	
	#Restriccion 8
	for j in range(K,N+K):
		row8 = []
		val8 = []
		row8.append(u_vars[j-1])
		val8.append(1.0)
		row8.append(x_vars[0,j])
		val8.append(-float(M))
		Model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = row8, val= val8)], senses = 'G', rhs = [float(t[0][j]-M)])

	#Restriccion 9
	for i in range(K, N+K):
		row9 = []
		val9 = []
		row9.append(u_vars[i-1])
		val9.append(1.0)
		Model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = row9, val= val9)], senses = 'G', rhs = [float(o[i])])

	#Restriccion 10
	for i in range(K,N+K):
		row10=[]
		val10=[]
		row10.append(u_vars[i-1])
		val10.append(1.0)
		Model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = row10, val= val10)], senses = 'L', rhs = [float(e[i])])
	
	#Restriccion 11
	for i in range(K,N+K):
		for j in range(K,N+K):
			row11 = []
			val11 = []
			if i!=j:
				row11.append(a_vars[i-K])
				val11.append(1.0)
				row11.append(u_vars[j-1])
				val11.append(1.0)
				row11.append(u_vars[i-1])
				val11.append(-1.0)
				row11.append(a_vars[j-K])
				val11.append(-1.0)						
				row11.append(x_vars[i,j])
				val11.append(float(M))		
				Model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = row11, val= val11)], senses = 'L', rhs = [float(M)])

	#Restriccion 13 
	for d in range(K):
		for j in range(K,N+K):
			row12 = []
			val12 = []
			row12.append(a_vars[j-K])
			val12.append(1.0)
			row12.append(x_vars[d,j])
			val12.append(-float(t[d,j]))
			Model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = row12, val= val12)], senses = 'G', rhs = [0.0])

	#Restriccion 14
	for i in range(K,N+K):
		row14=[]
		val14=[]
		row14.append(a_vars[i-K])
		val14.append(1.0)
		Model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = row14, val= val14)], senses = 'L', rhs = [float(tmax)])

	#Restriccion 15
	for i in range(K,N+K):
		row15=[]
		val15=[]
		row15.append(a_vars[i-K])
		val15.append(1.0)
		row15.append(u_vars[i-1])
		val15.append(-1.0)
		Model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = row15, val= val15)], senses = 'L', rhs = [0.0])

	#Restriccion 16
	for i in range(K,N+K):
		row19 = []
		val19 = []
		for j in range(N+K):
			if i!=j:
				row19.append(y_vars[j,i])
				val19.append(1.0)
		for j in range(N+K):
			if i!=j:
				row19.append(y_vars[i,j])
				val19.append(-1.0)
		Model.linear_constraints.add(lin_expr= [cplex.SparsePair(ind = row19, val = val19)], senses = 'E', rhs =[float(dem[i])])

	#Restriccion 17
	for i in range(N+K):
		for j in range(N+K):
			if i!=j:
				row16 = []
				val16 = []
				row16.append(y_vars[i,j])
				val16.append(1.0)
				row16.append(x_vars[i,j])
				val16.append(-float(Q))
				Model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = row16, val= val16)], senses = 'L', rhs = [0.0])

	#Setting Solver Parameters---------------------------------------------------------------------------------------------
	Model.objective.set_sense(Model.objective.sense.minimize)
	Model.parameters.timelimit.set(float(T_exec))
	Model.parameters.workmem.set(Memory)
	#Solving model---------------------------------------------------------------------------------------------------------
	tiemp_antes = Model.get_time()
	Model.solve()
	tiemp_despues = Model.get_time()
	tiemp_ejecucion=tiemp_despues - tiemp_antes
	def show_solution():
		x_sol =[]
		x_path=[]
		for i in range(N+K):
			for j in range(N+K):
				if(Model.solution.get_values("x("+str(i)+","+str(j)+")")!=0.0):
					x_sol.append("x("+str(i)+","+str(j)+")")
					x_path.append((i,j))
		u_sol = []
		for i in range(1,N+K):
			if (Model.solution.get_values('u('+str(i)+')') != 0.0):
				u_sol.append('u('+str(i)+')')
		a_sol = []
		for i in range(K,N+K):
			if (Model.solution.get_values('a('+str(i)+')') !=0.0):
				a_sol.append('a('+str(i)+')')
		return x_sol, x_path, u_sol, a_sol
	
	if Model.solution.get_status() == 101:
		estado = 'OPTIMO'
		obj_function = Model.solution.get_objective_value()
		x_sol, x_path, u_sol, a_sol = show_solution()
	elif Model.solution.get_status() == 107:
		estado = Model.solution.status[Model.solution.get_status()]
		obj_function = Model.solution.MIP.get_mip_relative_gap()
		x_sol,x_path = [0,0]
	else:
		estado = Model.solution.status[Model.solution.get_status()]
		obj_function = 0
		x_sol,x_path = [0,0]

	return estado, obj_function, x_sol,x_path, tiemp_ejecucion			

def plotear(x_path,K,N,x,y):
	G=nx.DiGraph()
	nodos = [i for i in range(N+K)]
	G.add_nodes_from(nodos)
	G.add_edges_from(x_path)
	depositos = [i for i in range(K)]
	pos = {}
	for i in range(N+K):
		if i < K:
			pos[i] = (x[i]+7.5*i,y[i])
		else:
			pos[i]= (x[i],y[i])
	#pos = {i: (x[i],y[i]) for i in range(K,N+K)}
	#pos = nx.spring_layout(G,pos)
	nx.draw_networkx_nodes(G, pos, nodelist = depositos , node_color = 'orange', node_size = 500, node_shape= 's')
	nx.draw_networkx_nodes(G, pos, nodelist= [i for i in nodos if i not in depositos],cmap=plt.get_cmap('jet'), node_color = 'c', node_size = 500)
	nx.draw_networkx_labels(G, pos)
	nx.draw_networkx_edges(G, pos, edgelist=x_path, arrows=True)
	#nx.draw(G, with_labels = True)
	#plt.savefig("Solution.png")
	plt.axis('off')
	plt.show()

def escribrir(instancia, estado, obj_function, x_sol, tiemp_ejecucion):
	"""
	Esta función toma los strings motor, solucion, x_sol, tiemp_ejecutado, y si plot es True grafica la sol
	primera columna: instancia, segunda: función objetivo y tercera: tiemp_ejecucion
	"""	
	archivo = open('resultados.txt','a')
	archivo.write(str(instancia)+';'+str(estado)+';'+str(obj_function)+';'+str(tiemp_ejecucion)+';\n')
	archivo.close()
	#enviar_email('resultados.txt')

def correr_uno(Clientes, rutas, tmax, num_instancia, tipo_instancia):
	if num_instancia <= 9:
		name = tipo_instancia+'20'+str(num_instancia)
		instancia = tipo_instancia+'20'+str(num_instancia)+'-'+str(Clientes)+'-t'+str(tmax)+'-R'+str(rutas)
	else:
		name = tipo_instancia+'2'+str(num_instancia)
		instancia = tipo_instancia+'2'+str(num_instancia)+'-'+str(Clientes)+'-t'+str(tmax)+'-R'+str(rutas)
	Q, c, t, o, e, d, s, x, y = crear_dat(Clientes,name,tmax,rutas)
	estado, obj_function, x_sol, x_path, tiemp_ejecucion = resolver_modelo(Clientes,rutas,Q,c,o,e,d,s,t,tmax, T_exec, Memory)
	print('Estado:',estado,'Tiempo Ejecución:', tiemp_ejecucion)
	if estado=='OPTIMO':
		print('Función Objetivo:', obj_function)
		print('SOOOOOOOOOOL:',x_sol)
		plotear(x_path,rutas, Clientes,x,y)
	escribrir(instancia, estado, obj_function, x_sol, tiemp_ejecucion)

T_exec = 10800 #Tiempo de ejecución máximo = a 3 horas por ahora
Memory = 4096 #Memoria RAM que le entregamos a Cplex para correr modelos, set a 4 GB

##############Prueba
correr_uno(Clientes=5, rutas=3, tmax=200, num_instancia=1, tipo_instancia='c')