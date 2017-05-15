#!/usr/bin/python
import sys
import argparse
f=sys.stdin
parser=argparse.ArgumentParser('Grand Plotting script')
parser.add_argument('-t','--title',help='plot title',type=str)
parser.add_argument('-o','--output',help='output into file, file type defined by its extention',type=str)
parser.add_argument('-n','--name',help='table got a head with names', action='store_true',default=False)
parser.add_argument('-p','--print',help='print value in position', action='store_true',default=False)
parser.add_argument('-b','--colorbar',help='show colorbar', action='store_true',default=False)
parser.add_argument('-N','--rowname',help='table first column with rownames', action='store_true',default=False)
parser.add_argument('-l','--log',help='log scale data (e,10 or 2)',type=str, choices=('10','e','2'),default='0')
parser.add_argument('-k','--keeppos',help='add a value to keep positive for log',type=float, default=0)
parser.add_argument('-r','--res',help='plot resolution',type=int, default=300)
parser.add_argument('-a','--plot_args',help='any plotting argument that can be passed to the corresponding plotting function',metavar='arg',nargs='+', type=str,default='')
args=vars(parser.parse_args())

d=[]
if args['name']:
	if args['rowname']:
		colname=f.readline().split()[1:]
	else:
		colname=f.readline().split()

rowname=[]
for l in f:
	ls=l.split()
	if args['rowname']:
		rowname.append(ls[0])
		d.append(map(float,ls[1:]))
	else:
		d.append(map(float,ls))
	
import numpy as np
d=np.array(d)

if args['log']!='0':
	logdic={'10':np.log(10),'2':np.log(2),'e':1.}
	d=np.log(d+args['keeppos'])/logdic[args['log']]

import matplotlib.pyplot as plt

fig=plt.figure()
ax=plt.subplot(111)
ax.set_xlim(0,d.shape[1])
ax.set_ylim(0,d.shape[0])

d = np.ma.masked_invalid(d)

cax=ax.pcolor(d)
ax.patch.set(hatch='x', edgecolor='black')

if args['colorbar']:
	fig.colorbar(cax)
if args['print']:
	for i in xrange(len(d)):
		for j in xrange(len(d[0])):
			ax.text(j+.5,i+.5,int(d[i,j]),color='w',ha='center',va='center',fontsize=20)

if args['name']:
	ax.set_xticks(np.arange(len(colname))+0.5)
	ax.set_xticklabels(colname,rotation='vertical',fontsize=20)
if args['rowname']:
	ax.set_yticks(np.arange(len(rowname))+0.5)
	ax.set_yticklabels(rowname,fontsize=20)

if args['output']:
	plt.savefig(args['output'],dpi = args['res'])
else:
	plt.show()

