import cond_grad as cd
funct = cd.Functional(lambda x: ((x[0]**2)-(x[1]**4) + x[2])/((x[2]**5)+(4)),
	                  [[1.0, 3.0], [3.0, 5.0], [4.0, 8.0]], 3,
	                  "F = (x_0^2 - x_1^4)/(x_2^5 + 4)", 0)
funct.Optimize()