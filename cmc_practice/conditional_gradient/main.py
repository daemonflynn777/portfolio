import cond_grad as cd
import numpy as np
from tkinter import *
from tkinter.ttk import Checkbutton
from tkinter import scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from math import sqrt

def clicked():
	#start_point = np.array([np.random.uniform(input_lims[i][0], input_lims[i][1], 1) for i in range(dimensions)]).reshape(dimensions)
	#start_point = [1.0, 1.0, 1.0]
	ax.cla()
	ax.grid()
	graph.draw()
	if len(txt2.get()) != 0:
		center = list(map(float, txt2.get().split()))
	else:
		center = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
	if len(txt3.get()) != 0:
		radius = float(txt3.get())
	else:
		radius = 363
	print(radius)
	start_point = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
	#start_point = np.array([np.random.uniform(sqrt(2*(radius))/2.0, 1) for i in range(len(center))]).reshape(len(center))
	funct = cd.Functional(lambda x: 150*((x[1] - x[0])**4 + (x[2] - 2*x[0])**4 + (x[3] - 3*x[0])**4 + (x[4] - 4*x[0])**4 + (x[5] - 5*x[0])**4) + (x[0] - 2)**4,
	                  center, 363, start_point,
	                  func_to_str, silent = 0)
	scroll_txt.delete(1.0, END)
	funct.Optimize(scroll_txt, ax, graph)

func_to_str = "J(x) = 150*sum_(i=2)^(6)(x(i)-ix(1))^4+(x(1)-2)^4"
window = Tk()
window["bg"] = 'dim gray'
window.title("Метод условного градиента")
window.geometry('1500x620')
lbl1_1 = Label(window, text = "Минимизируемый функционал:", font = ("Arial", 24))
lbl1_1.configure(background = 'dim gray')
lbl1_1.grid(column = 0, row = 0)
lbl1_2 = Label(window, text = func_to_str, font = ("Arial", 22))
lbl1_2.configure(background = 'dim gray')
lbl1_2.grid(column = 0, row = 1)

lbl2 = Label(window, text = "Точность", font = ("Arial", 14))
lbl2.grid(column = 1, row = 0, sticky = 'W')
txt2 = Entry(window, width = 15)
txt2.configure(background = 'LightSteelBlue4')
txt2.grid(column = 1, row = 0)

lbl3 = Label(window, text = "Макс. итерации", font = ("Arial", 14))
lbl3.grid(column = 1, row = 1, sticky = 'W')
txt3 = Entry(window, width = 15)
txt3.configure(background = 'LightSteelBlue4')
txt3.grid(column = 1, row = 1)

lbl4 = Label(window, text = "Начальная точка", font = ("Arial", 14))
lbl4.grid(column = 1, row = 2, sticky = 'W')
txt4 = Entry(window, width = 15)
txt4.configure(background = 'LightSteelBlue4')
txt4.grid(column = 1, row = 2)
chk4_state = BooleanVar()
chk4_state.set(True)
chk4 = Checkbutton(window, text = "Задать автоматически", var = chk4_state)
chk4.grid(column = 1, row = 2, sticky = 'E')

scroll_txt = scrolledtext.ScrolledText(window, width = 100, height = 27, pady = 5, padx = 5)
scroll_txt.configure(background = 'LightSteelBlue4')
scroll_txt.grid(column = 0, row = 3)

fig = Figure()     
ax = fig.add_subplot(111) 
ax.set_xlabel("X axis") 
ax.set_ylabel("Y axis") 
ax.grid()

graph = FigureCanvasTkAgg(fig, master = window)
graph.get_tk_widget().grid(column = 1, row = 3, pady = 5, padx = 5)

btn = Button(window, text = "Минимизировать", bg = "grey", command = clicked)
btn.configure(background = 'LightSteelBlue4')
btn.grid(column = 0, row = 2)

window.mainloop()

