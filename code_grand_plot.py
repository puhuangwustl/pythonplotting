#!/usr/bin/python

# this code is written for simple histogram or scatter plot, given input is STDIN with simple tabulate format
# it is written to be able to conveniently pileup with the shell commands such as 'grep', 'cut', 'awk','pdftk' etc.
# for fast visualizing the data in required ways, also for storing
# read only the first two fields, if more than two fields as input
# e.g. grep -P "scaffold_1\t" ../gbs/pilot_data/sorted/tags/multicov_avg10.bed | cut -f2,4 | ./code_grand_plot.py -t hist -a "color='r'" bins=50 -o out1.pdf
#      grep -P "scaffold_1\t" ../gbs/pilot_data/sorted/tags/multicov_avg10.bed | cut -f2,4 | ./code_grand_plot.py -a "color='g'" -o out2.pdf
#      pdftk out1.pdf out2.pdf cat output combine.pdf


import argparse
import sys
import numpy as np


parser=argparse.ArgumentParser('Grand Plotting script')
parser.add_argument('-t','--type',help='plot type: {xy, line, hist, bar, barstack, bar3d}',choices=('xy','line','hist','bar', 'barstack', 'bar3d'),type=str,default='xy')
parser.add_argument('--title',help='plot title',type=str)
parser.add_argument('-o','--output',help='output into file, file type defined by its extention',type=str)
parser.add_argument('-n','--name',help='table got a head with names', action='store_true',default=False)
parser.add_argument('-N','--rowname',help='table first column with rownames', action='store_true',default=False)
parser.add_argument('-l','--log',help='log scale the y axis (e,10 or 2)',type=str, choices=('10','e','2'),default='0')
parser.add_argument('-r','--regression',help='do regression analysis',action='store_true',default=False)
parser.add_argument('-d','--density',help='do density heat coloring (may take long time for large datasets)',action='store_true',default=False)
parser.add_argument('-a','--plot_args',help='any plotting argument that can be passed to the corresponding plotting function',metavar='arg',nargs='+', type=str,default='')
parser.add_argument('--xlim',help='set xlim, turple of 2',type=float, nargs=2)
parser.add_argument('--ylim',help='set ylim, turple of 2',type=float, nargs=2)
args=vars(parser.parse_args())
f=sys.stdin
if args['name']:
	if args['rowname']:
		head=f.readline().split()[1:]
	else:
		head=f.readline().split()
#print args['plot_args']

data=[]
if args['rowname']:
	rowname=[]
	for l in f:
		rowname.append(l.split()[0])
		data.append(map(float,l.split()[1:]))
else:
	for l in f:
		data.append(map(float,l.split()))
if args['rowname']:
	rowname=np.array(rowname)
else:
	rowname=''
data=map(list,zip(*data))
nvariable=len(data)

def transform(x,logscale):
	dic={'0':'','10':'np.log10','e':'np.log','2':'np.log2'}
	exec 'xx='+dic[logscale]+'(x)'
	return np.array(xx)

data=transform(data,args['log'])

if args['regression']:
	from scipy import stats

import matplotlib as mpl
if (args['output']):
	mpl.use('Agg')
import matplotlib.pyplot as plt

if args['plot_args']!='':
	plotarg=", ".join(args['plot_args'])+','
else:
	plotarg=""

fig=plt.figure(0)

def call_xyplot(x,y, rowname):
	if args['name']:
		plt.xlabel(head[0])
		plt.ylabel(head[1])
	if args['density']:
		from scipy.stats import gaussian_kde
		xy = np.vstack([x,y])
		z = gaussian_kde(xy)(xy)
		# Sort the points by density, so that the densest points are plotted last
		idx = z.argsort()
		if args['rowname']:
			x, y, z, rowname = x[idx], y[idx], np.log10(z[idx]), rowname[idx]
		else:
			x, y, z = x[idx], y[idx], np.log10(z[idx])
		exec "plt.scatter(x,y,"+plotarg+" c=z, edgecolor='')"
	else:
		exec "plt.plot(x,y, "+plotarg+" )"
	if args['regression']:
		slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
		print 'slope:',slope, 'intercept:',intercept,'r:',r_value,'p:',p_value
		line=[]
		xsort=np.sort(x)
		for i in xsort:
			line.append(slope*i+intercept)
		plt.plot(xsort,line,'-')

if (nvariable==1 and args['type']=='xy') or args['type']=='line':
	if args['name']:
		head.insert(0,'FlowNumber')
	x=range(len(data[0]))
	for y in data:
		call_xyplot(x,y,rowname)

elif args['type']=='xy':
	x=data[0]
	for y in data[1:]:
		call_xyplot(x,y,rowname)

elif args['type']=='hist':
	if args['name']:
		xlab=','.join(head)
		plt.xlabel(xlab)
	if args['density']:
		from scipy.stats.kde import gaussian_kde
		for x in data:
			pdf=gaussian_kde(x)
			xsmooth=np.linspace(min(x),max(x),max(len(x)*4,200))
			plt.plot(xsmooth,pdf(xsmooth),lw=2)
		exec "plt.hist(data.tolist(), "+plotarg+" normed=1 )"
	else:
		exec "plt.hist(data.tolist(), "+plotarg+" )"

elif args['type']=='barstack':
        if args['name']:
                head.insert(0,'FlowNumber')
        x=range(len(data[0]))
	c=['r','g','b','yellow','purple']
	b=np.array([0 for i in x])
	i=0
	for y in data:
		#plt.barh(x,y,left=b,color=c[i%len(c)])
		exec "plt.bar(x,y,bottom=b,color=c[i%len(c)], "+plotarg+" )"
		b=b+y
		i+=1

elif args['type']=='bar':
        if args['name']:
                head.insert(0,'FlowNumber')
        x=np.array(range(len(data[0])))
	c=['r','g','b','yellow','purple']
	barwid,i=0.8/len(data),0
	for y in data:
		x=x+barwid
		exec "plt.bar(x,y,width=barwid,color=c[i%len(c)], "+plotarg+" )"
		i+=1

elif args['type']=='bar3d':
	from mpl_toolkits.mplot3d import Axes3D
        if args['name']:
                head.insert(0,'FlowNumber')
        ax = fig.add_subplot(111, projection='3d')
	#ax.pbaspect = [2.0, 0.6, 0.25]
	lx,ly=len(data[0]),len(data)
	m=max(lx,ly)
	dx,dy=0.6,0.6
	disx=m/1./lx
	disy=m/1./ly
	x=np.array(range(lx))*disx+disx/2-dx/2.
	z=np.array([0 for i in range(lx)])
	y=z*disy+disy/2-dy/2
	c=['r','g','b','yellow','purple']
	i=0
	for dz in data:
		exec "ax.bar3d(x,y,z,dx,dy,dz=dz, color=c[i%len(c)], "+plotarg+")"
		y=y+disy
		i+=1
	ax.auto_scale_xyz([0, m], [0, m], [0,np.max(data)*1.1])
	#ax.auto_scale_xyz([0, lx+1], [0, y[0]], [0,max(data)*1.1])

if args['rowname']:
	for i in xrange(len(rowname)):
		if rowname[i]!='NULL':
			plt.annotate(rowname[i],(x[i],y[i]),(x[i]+0.1,y[i]+0.1))

if args['xlim']:
	plt.xlim(args['xlim'])
if args['ylim']:
	plt.ylim(args['ylim'])

if args['title']:
	plt.title(args['title'])

if (args['output']):
	fig.savefig(args['output'])
else:
	plt.show(block=True)

