import cond_grad as cd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import pdfkit as pdf
#from PyPDF2 import PdfFileMerger

# M A I N
clmns = [f'x_start{i}' for i in range(6)] + [f'x_end_{i}' for i in range(6)] + ['f(x_end)'] + ['iter_count']
research_data = []

for i in range(-7, 8):
	center = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
	radius = 363
	start_point = np.full(6, float(i))
	precision = 0.01
	m_iter = 1000
	func_to_str = "J(x) = 150*sum_(i=2)^(6)(x(i)-ix(1))^4+(x(1)-2)^4"
	funct = cd.Functional(lambda x: 150*((x[1] - x[0])**4 + (x[2] - 2*x[0])**4 + (x[3] - 3*x[0])**4 + (x[4] - 4*x[0])**4 + (x[5] - 5*x[0])**4) + (x[0] - 2)**4,
		                  center, radius, start_point, precision, m_iter,
		                  func_to_str, silent = 0)
	dots_set = funct.Optimize()
	research_data.append(dots_set)
	print(dots_set)

research_df = pd.DataFrame(research_data, columns = clmns)
research_df.to_excel("research.xlsx")
#pdf.from_file("table.html", "table.pdf")

#research_results = PdfPages('research.pdf')
#fig, ax =plt.subplots(figsize=(12,4))
#ax.axis('tight')
#ax.axis('off')
#the_table = ax.table(cellText=df.values,colLabels=df.columns,loc='center')

fig = plt.figure(figsize = (16, 9), num = "Зависимость кол-ва итераций от выбора начальной точки")
plt.plot(range(-7, 8), research_df['iter_count'])
plt.title("Зависимость кол-ва итераций от выбора начальной точки")
plt.xlabel("Значение x в начальном векторе (x, x, x, x, x, x)")
plt.ylabel("Кол-во итераций")
plt.show()
plt.close()
plt.savefig("research_plot.png", dpi = 300)

#pdf_merger = PdfFileMerger()
#pdf_merger.append("table.pdf")
#pdf_merger.append("plot.pdf")
#research_file = open("research.pdf", 'wb')
#pdf_merger.write(research_file)
