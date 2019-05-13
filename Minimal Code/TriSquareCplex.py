# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 12:28:49 2015

@author: bastinux
"""
###############################################################################################################
###               Notes: Program requires CDTSquare.1.  Node, edge and ele (face) files;                    ###
### Program creates Triangle Edge Boundary Matrix and uses input 1-chain to find optimized homologous chain ###
###############################################################################################################

from numpy import *
import sys
from scipy.sparse import lil_matrix
from scipy import linalg
import scipy.spatial.distance
orig_stdout = sys.stdout

##############################################################################
###    Import Count information; Call lists, matrix, and set Maxfaces      ###
##############################################################################

NumNodes = int(open('CDTSquare.1.node').readlines()[0].split()[0]);
NumEdges = int(open('CDTSquare.1.edge').readlines()[0].split()[0]);
NumFaces = int(open('CDTSquare.1.ele').readlines()[0].split()[0]);

MaxFaces = 12;

Faces = {};
FaceOrients = {};
Edges = {};
EdgeIndices = {};
Nodes = {};
TriEdgeBoundary=lil_matrix((NumEdges, NumFaces));

##############################################################################
###             Import Node Information into dictionay format              ###
##############################################################################
###    Stores vertices by vertex number as [float_1, float_2, float_3]     ###
##############################################################################

### Read in the first line split into array of three elements, store as list ###

line1 = open('CDTSquare.1.node').readlines()[1].split();
node1 = [float(line1[1]),float(line1[2]),0];
Nodes[1] = [node1[0],node1[1],node1[2]];
i=1;

### Read in the remaining lines split into array of three elements, store as list and dictionary###

with open('CDTSquare.1.node') as f:
  for line in f.readlines()[2:NumNodes+1]:
    i=i+1;
    Nodes[i] = [float(line.split()[1]),float(line.split()[2]),float(0)];

##############################################################################
###             Import Edge Information into dictionay format              ###
##############################################################################
###      Stores edges by sorted Vertex v_1,v_2 as [edgenumber, distance]   ###
##############################################################################

### Read in the first line, split into array of three elements, sort vertices, calc dist then store as list and dictionary ###

line1 = open('CDTSquare.1.edge').readlines()[1].split();
edge1 = sorted([int(line1[1]),int(line1[2])]);
Len = scipy.spatial.distance.pdist([array(Nodes[edge1[0]]),array(Nodes[edge1[1]])]);
EdgeIndices[edge1[0],edge1[1]] = [1];
Edges[1]=[edge1[0],edge1[1],Len[0]];
i=1;

### Read in the remaining lines split into array of three elements, sort vertices, calc dist then store as list and dictionary ###

with open('CDTSquare.1.edge') as f:
  for line in f.readlines()[2:NumEdges+1]:
    i=i+1;
    edge = sorted([int(line.split()[1]),int(line.split()[2])]);
    Len = scipy.spatial.distance.pdist([array(Nodes[edge[0]]),array(Nodes[edge[1]])]);
    EdgeIndices[edge[0],edge[1]] = [i];
    Edges[i]=[edge[0],edge[1],Len[0]];

##############################################################################################################################################
###                            Import Face Orientation and Area Information into dictionary format                                         ###
##############################################################################################################################################
###    Stores FaceOrients by sorted Vertex v_1,v_2,v3 as [facenumber, orientation (1 if agrees with lexographic ordering, -1 else), area]  ###
##############################################################################################################################################

### Read in the first line, split into array of four elements, sort vertices, calc area then store as list and dictionary. Also store FaceOrientations commented out ###

line1 = open('CDTSquare.1.ele').readlines()[1].split();
face1 = sorted([int(line1[1]),int(line1[2]),int(line1[3])]);
face1area = absolute(linalg.det(array([[1,1,1],array(Nodes[face1[1]])-array(Nodes[face1[0]]),array(Nodes[face1[2]])-array(Nodes[face1[0]])]))/2.0);
# FaceOrients[face1[0],face1[1],face1[2]]=[1,1];
# FaceOrients[face1[1],face1[2],face1[0]]=[1,1];
# FaceOrients[face1[2],face1[0],face1[1]]=[1,1];
# FaceOrients[face1[0],face1[2],face1[1]]=[1,-1];
# FaceOrients[face1[2],face1[1],face1[0]]=[1,-1];
# FaceOrients[face1[1],face1[0],face1[2]]=[1,-1];
Faces[1]=[face1[0],face1[1],face1[2],face1area];
i=1;
TriEdgeBoundary[EdgeIndices[face1[0],face1[1]][0]-1,i-1]=1;
TriEdgeBoundary[EdgeIndices[face1[1],face1[2]][0]-1,i-1]=1;
TriEdgeBoundary[EdgeIndices[face1[0],face1[2]][0]-1,i-1]=-1;

### Read in the remaining lines, split into array of four elements, sort vertices, calc area then store as list and dictionary. Also store FaceOrientations commented out ###

with open('CDTSquare.1.ele') as f:
    for line in f.readlines()[2:NumFaces+1]:
      i=i+1;
      face = sorted([int(line.split()[1]),int(line.split()[2]),int(line.split()[3])]);
      facearea = absolute(linalg.det(array([[1,1,1],array(Nodes[face[1]])-array(Nodes[face[0]]),array(Nodes[face[2]])-array(Nodes[face[0]])]))/2.0);
#       FaceOrients[face[0],face[1],face[2]]=[i,1];
#       FaceOrients[face[1],face[2],face[0]]=[i,1];
#       FaceOrients[face[2],face[0],face[1]]=[i,1];
#       FaceOrients[face[0],face[2],face[1]]=[i,-1];
#       FaceOrients[face[2],face[1],face[0]]=[i,-1];
#       FaceOrients[face[1],face[0],face[2]]=[i,-1];
      Faces[i]=[face[0],face[1],face[2],facearea];
      TriEdgeBoundary[EdgeIndices[face[0],face[1]][0]-1,i-1]=1;
      TriEdgeBoundary[EdgeIndices[face[1],face[2]][0]-1,i-1]=1;
      TriEdgeBoundary[EdgeIndices[face[0],face[2]][0]-1,i-1]=-1;

TriEdgeBoundary=TriEdgeBoundary.tocsr();

####################################################################################################
###                                     Cplex                                                    ###
####################################################################################################

###########################################################################
####     Main Problem Setup: Objective Statement maximize edgedists    ####
###########################################################################
####   max u^T*z with u=edgedists,edgedists,0*facedists, 0*facedists   ####
###########################################################################

import cplex
from cplex.exceptions import CplexError
import sys

prob = cplex.Cplex();
prob.objective.set_sense(prob.objective.sense.maximize);

edgedists = zeros(NumEdges);
for i in range(0,NumEdges):
  edgedists[i]=Edges[i+1][2];
edgedists = append(edgedists,edgedists);
zerofaces = zeros(NumFaces*2);
z = append(edgedists,zerofaces);

my_obj = z;
prob.variables.add(obj = my_obj, types = [prob.variables.type.binary]*len(my_obj));

###########################################################################
####                  Main Constraints: Homology constraint            ####
###########################################################################

### Set c to be chain off lower main edge ###

c = zeros(NumEdges);
for i in range(0,34):
  c[EdgeIndices[i+1,i+2][0]-1]=1;
my_rhs = c;

prob.linear_constraints.add(rhs=my_rhs);

I= scipy.sparse.identity(NumEdges);
A = scipy.sparse.hstack([I,-I,-TriEdgeBoundary, TriEdgeBoundary]);

for i, j, s in zip(A.row, A.col, A.data):
  x=int(i);
  y=int(j);
  v=int(s);
  prob.linear_constraints.set_coefficients(x, y, v);

################################################################################################################
####      Additional Constraints:      Binary Edge Direction, Binary Face Choice and FaceCount              ####
################################################################################################################
####      Bz<=1 gives binary edge choice, [00,...,0,1,1,...,1]*z<= maxfaces, Cz<=1 gives binary face choice ####
################################################################################################################

##  Edge Choice ##

ZeroBoundary = zeros((shape(TriEdgeBoundary)));
B = scipy.sparse.hstack([I,I,ZeroBoundary, ZeroBoundary]);

my_rhs2 = ones(NumEdges);
prob.linear_constraints.add(rhs = my_rhs2);

for i, j, s in zip(B.row, B.col, B.data):
  x=int(i)+NumEdges;
  y=int(j);
  v=int(s);
  prob.linear_constraints.set_coefficients(x, y, v);
  prob.linear_constraints.set_senses(x, "L");

##  Maxfaces  ##

zeroedges = zeros(NumEdges*2);
facecount = ones(NumFaces*2);
facecount = append(zeroedges, facecount);
my_rhs3 = ones(1);
my_rhs3[0] = MaxFaces;

prob.linear_constraints.add(rhs = my_rhs3);

for i in range(0,NumFaces*2):
    x=NumEdges*2;
    y=i+NumEdges*2;
    v=1;
    prob.linear_constraints.set_coefficients(x, y, v);
prob.linear_constraints.set_senses(x, "L");

##  Face Choice  ##

I = scipy.sparse.identity(NumFaces);
C = scipy.sparse.hstack([I,I]);

my_rhs4 = ones(NumFaces);
prob.linear_constraints.add(rhs = my_rhs4);

for i, j, s in zip(C.row, C.col, C.data):
  x=int(i)+NumEdges*2+1;
  y=int(j)+NumEdges*2;
  v=int(s);
  prob.linear_constraints.set_coefficients(x, y, v);
  prob.linear_constraints.set_senses(x, "L");



###########################################################################################
####      Solve, calculate edgepath and facepath: print out solution against original  ####
###########################################################################################


prob.solve();

status = prob.solution.get_status()
objval = prob.solution.get_objective_value()

z = prob.solution.get_values()

EdgePath=[];
FacePath=[];

##  EdgePath  ##

for i in range(0,2*NumEdges):
  if i < NumEdges:
    EdgePath.append(z[i]);
  if i >= NumEdges:
    EdgePath[i-NumEdges]=EdgePath[i-NumEdges]-z[i];

##  FacePath  ##

for i in range(2*NumEdges,len(z)):
  if i < 2*NumEdges+NumFaces:
    FacePath.append(z[i]);
  if i >= 2*NumEdges+NumFaces:
    FacePath[i-(2*NumEdges+NumFaces)]=FacePath[i-(2*NumEdges+NumFaces)]-z[i];

##  Homologous Test  ##

cprime = A*z;
for i in range(0,len(EdgeIndices)):
  if cprime[i] != c[i]:
    print "Not Homologous";

###################################################################################################################
########              Write Path Solution to Showme readable file and print solution            ###################
###################################################################################################################

## Write poly file ##

f = file('CDTSquare.2.poly','w')
sys.stdout = f

with open('CDTSquare.1.node') as f:
  for line in f.readlines():
    print line

print count_nonzero(EdgePath), 1;

j=0;

for i in range(0,NumEdges):
  if EdgePath[i] != 0:
    j = j+1;
    print j, Edges[i+1][0], Edges[i+1][1], 1;

print 0;
sys.stdout = orig_stdout
f.close()

## Write node file ##

f = file('CDTSquare.2.node','w')
sys.stdout = f

with open('CDTSquare.1.node') as f:
  for line in f.readlines():
    print line

print 0;
sys.stdout = orig_stdout
f.close()

## Write ele file ##

f = file('CDTSquare.2.ele','w')
sys.stdout = f
print count_nonzero(FacePath), 3, 1;
j=0;

for i in range(0,NumFaces):
  if FacePath[i] != 0:
    j = j+1;
    print j, Faces[i+1][0], Faces[i+1][1], Faces[i+1][2], 1;

print 0;
sys.stdout = orig_stdout
f.close()

## Print Solution set ##

print "Solution is ", z;
print "Solution EdgePath is ", EdgePath, "with length", dot(my_obj,z);
print "Originial path is", c, "with length", dot(edgedists[0:NumEdges],absolute(c));

print "Solution Edge list is ";
for i in range(0,len(EdgePath)):
  if EdgePath[i] != 0:
    print Edges[i+1][0], Edges[i+1][1], 0;

print "Original Edge list is ";
for i in range(0,len(EdgePath)):
  if c[i] != 0:
    print Edges[i+1][0], Edges[i+1][1], 1;

####################################################################################################################################################################
## End Main ##
####################################################################################################################################################################

#### Extras and Checks ######

# print(prob.variables.get_types());
# print(prob.variables.get_num());
# print(prob.linear_constraints.get_num());
# print prob.linear_constraints.get_senses();
# print prob.linear_constraints.get_rhs();
# print TriEdgeBoundary ;
# print shape(TriEdgeBoundary);
# print len(Edges);
# print TriEdgeBoundary[len(Edges)-1,len(Faces)-1];
# print 'Number of nonzero entries in TriEdgeBoundary Matrix =' ,TriEdgeBoundary.nnz;
# print TriEdgeBoundary.nnz/3;
# print NumFaces;
# print(len(FaceOrients));
# print(len(EdgeIndices));
#print"Number of FaceOrients in Boundary of Mesh is", NumFaces;
#print'Number of EdgeIndices in full Mesh is ', NumEdges;
#print(FaceOrients)
#print(EdgeIndices[(183, 50)])
# print Edges;

#if FaceOrients.get((face1[0],face1[1],face1[2]), "Interior") != "Interior":
#  print "Boundary Triangle";

# faceareasw = zeros(NumFaces);
# for i in range(0,NumFaces):
#   faceareasw[i]=Faces[i+1][3]*.01;
# faceareasw = append(faceareasw,faceareasw);
