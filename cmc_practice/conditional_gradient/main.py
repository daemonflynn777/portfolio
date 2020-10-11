import cond_grad as cd
funct = cd.Functional(lambda x: x[0]**3 + (x[0]**2)/(x[1]**2) + x[1]**3,
	                  lambda y: 2*(y**3) + 1,
	                  [[-1.0, 3.0], [3.0, 5.0]], 2,
	                  "F = x_0^3 + x_0^2/x_1^2 + x_1^3", 0)
funct.Optimize()