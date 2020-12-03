import numpy as np
import sympy as sp
import math
from sympy import sin, cos, acos, nsimplify, pprint
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tabulate import tabulate
from sympy import init_printing
# init_printing()

def cpprint(string, matrix):
    print(string)
    # for pos in matrix:
    #     if pos<1e-10:
    #         pos = 0
    #     pos = round(pos,10)
    pprint(matrix.evalf(10), use_unicode=False)

# Symbolic
a10, a11, a12, a13, a14, a15, a1, a2, a3, a4, a5, a6 = sp.symbols('a10 a11 a12 a13 a14 a15 a1 a2 a3 a4 a5 a6')
th1, th2, th3, th4, th5, th6, d1, d2, d3, d4, d5, d6 = sp.symbols('th1 th2 th3 th4 th5 th6 d1 d2 d3 d4 d5 d6')
L1, L2, L3, L4, Lc1, Lc2, Lc3, Lc4, Lc5, Lc6, g = sp.symbols('L1 L2 L3 L4 Lc1 Lc2 Lc3 Lc4 Lc5 Lc6 g')
th1_p, th2_p, th3_p, th4_p, th5_p, th6_p, d1_p, d2_p, d3_p, d4_p, d5_p, d6_p = sp.symbols(
    'th1_p th2_p th3_p th4_p th5_p th6_p d1_p d2_p d3_p d4_p d5_p d6_p')
th1_pp, th2_pp, th3_pp, th4_pp, th5_pp, th6_pp, d1_pp, d2_pp, d3_pp, d4_pp, d5_pp, d6_pp = sp.symbols(
    'th1_pp th2_pp th3_pp th4_pp th5_pp th6_pp d1_pp d2_pp d3_pp d4_pp d5_pp d6_pp')
om22x, om22y, om22z, om33x, om33y, om33z, om44x, om44y, om44z, om55x, om55y, om55z, om66x, om66y, om66z = sp.symbols(
    'om22x om22y om22z om33x om33y om33z om44x om44y om44z om55x om55y om55z om66x om66y om66z')
om22x_p, om22y_p, om22z_p, om33x_p, om33y_p, om33z_p, om44x_p, om44y_p, om44z_p, om55x_p, om55y_p, om55z_p, om66x_p, om66y_p, om66z_p = sp.symbols(
    'om22x_p om22y_p om22z_p om33x_p om33y_p om33z_p om44x_p om44y_p om44z_p om55x_p om55y_p om55z_p om66x_p om66y_p om66z_p')
vel22x, vel22y, vel22z, vel33x, vel33y, vel33z, vel44x, vel44y, vel44z, vel55x, vel55y, vel55z, vel66x, vel66y, vel66z = sp.symbols(
    'vel22x vel22y vel22z vel33x vel33y vel33z vel44x vel44y vel44z vel55x vel55y vel55z vel66x vel66y vel66z')
vel22x_p, vel22y_p, vel22z_p, vel33x_p, vel33y_p, vel33z_p, vel44x_p, vel44y_p, vel44z_p, vel55x_p, vel55y_p, vel55z_p, vel66x_p, vel66y_p, vel66z_p = sp.symbols(
    'vel22x_p vel22y_p vel22z_p vel33x_p vel33y_p vel33z_p vel44x_p vel44y_p vel44z_p vel55x_p vel55y_p vel55z_p vel66x_p vel66y_p vel66z_p')
a1_v = sp.Matrix([a10, a11, a12, a13, a14, a15])
a_v = sp.Matrix([a1, a2, a3, a4, a5, a6])
th_v = sp.Matrix([th1, th2, th3, th4, th5, th6])
d_v = sp.Matrix([d1, d2, d3, d4, d5, d6])

g = 9.81
# Numeric data from SCARA
# Length of the links
_L1_ = 0.75 # in meters
_L2_ = 0.40 # in meters
_L3_ = 0.40 # in meters
_a1_ = 0.15 # in meters
# L5 = # in meters
_a2_ = 0.40 # in meters

# Inverse kinematics
x = float(input('Coordenada x (en cm): '))
y = float(input('Coordenada y (en cm): '))
z = float(input('Coordenada z (en cm): '))
x/=100
y/=100
z/=100
Px = L2*cos(th1) + L3*cos(th1+th2)
Py = L2*sin(th1) + L3*sin(th1+th2)
Pz = L1+a1-d3-a2
print('Px =', Px)
print('Py =', Py)
print('Pz =', Pz)

q = (x**2 + y**2)**(1/2)
print('q = (x^2 + y^2)^(1/2) = ', q)
beta = acos((L2**2 + L3**2 - q**2)/(2*L2*L3))
beta_solr = beta.subs({L2: _L2_, L3: _L3_})
beta_sold = beta_solr*180/np.pi
print(f'{chr(946)} = arc cos[(L2^2 + L3^2 - q^2)/(2*L2*L3)] = ', beta, '=', beta_sold)
theta2r = np.pi - beta_solr
theta2d = theta2r*180/np.pi
print(f'{chr(952)}2 = 180° - {chr(946)} = ', theta2d)
alpha = acos((q**2 + L2**2 - L3**2 )/(2*L2*q))
alpha_solr = alpha.subs({L2: _L2_, L3: _L3_})
alpha_sold = alpha_solr*180/np.pi
print(f'{chr(945)} = arc tan[(q^2 + L2^2 - L3^2 )/(2*L2*q)] = ', alpha, '=', alpha_sold)
thetaqr = math.atan2(y,x)
thetaqd = thetaqr*180/np.pi
print(f'{chr(952)}q = arc tan(y/x) = ', thetaqd)
theta1r = thetaqr - alpha_solr
theta1d = theta1r*180/np.pi
print(f'{chr(952)}1 = {chr(952)}q - {chr(945)} = ', theta1d)
d3_ = L1+a1-a2-z
d3_sol = d3_.subs({L1: _L1_, a1: _a1_, a2: _a2_})
print('d3 = L1+a1-a2-Pz = ', d3_, '=', d3_sol)
print()
invKin = [[theta1r, theta2r, d3_sol*100],[theta1d, theta2d, d3_sol*100]]
print(tabulate(invKin, headers=[chr(952)+'1', chr(952)+'2', 'd3']))
print()

# Mass of the links
# mb = 317.072591 # in kilograms
m1 = 46.604773 # in kilograms
m2 = 0
m3 = 74.295959 # in kilograms
m4 = 0
m5 = 16.618563 # in kilogramos
m6 = 0
# Link1
Lc1x, Lc1y, Lc1z = 0.200000, 0, 0 # from solidworks
Ixx1, Iyy1, Izz1 = 0.282807, 1.375603, 1.353315 # from solidworks
Ic1 = sp.Matrix([
                [Ixx1,0,0],
                [0,Iyy1,0],
                [0,0,Izz1]])
# Link2
Lc2x, Lc2y, Lc2z = -0.200000, 0, 0 # from solidworks
Ixx2, Iyy2, Izz2 = 0.282807, 1.375603, 1.353315 # from solidworks
Ic2 = sp.Matrix([
                [Ixx2,0,0],
                [0,Iyy2,0],
                [0,0,Izz2]])
# Link3
Lc3x, Lc3y, Lc3z = 0.157849, 0, 0.001766 # from solidworks
Ixx3, Iyy3, Izz3 = 0.617768, 3.788118, 3.705766 # from solidworks
Ic3 = sp.Matrix([
                [Ixx3,0,0],
                [0,Iyy3,0],
                [0,0,Izz3]])
# Link4
Lc4x, Lc4y, Lc4z = -0.242151, 0, -0.001766 # from solidworks
Ixx4, Iyy4, Izz4 = 0.617768, 3.788118, 3.705766 # from solidworks
Ic4 = sp.Matrix([
                [Ixx4,0,0],
                [0,Iyy4,0],
                [0,0,Izz4]])
# Link5
Lc5x, Lc5y, Lc5z = 0.000017, -0.000095, -0.035844 # from solidworks
Ixx5, Iyy5, Izz5 = 1.208600, 1.208659, 0.012573 # from solidworks
Ic5 = sp.Matrix([
                [Ixx5,0,0],
                [0,Iyy5,0],
                [0,0,Izz5]])
# Link6
Lc6x, Lc6y, Lc6z = 0.000015, 0.000086, 0.436514 # from solidworks
Ixx6, Iyy6, Izz6 = 1.208600, 1.208659, 0.012573 # from solidworks
Ic6 = sp.Matrix([
                [Ixx6,0,0],
                [0,Iyy6,0],
                [0,0,Izz6]])
# Center of mass of the links
P1c1 = sp.Matrix([Lc1x,Lc1y,Lc1z])
P2c2 = sp.Matrix([Lc2x,Lc2y,Lc2z])
P3c3 = sp.Matrix([Lc3x,Lc3y,Lc3z])
P4c4 = sp.Matrix([Lc4x,Lc4y,Lc4z])
P5c5 = sp.Matrix([Lc5x,Lc5y,Lc5z])
P6c6 = sp.Matrix([Lc6x,Lc6y,Lc6z])

# # Example
# data = [
#     [1, 0, 0, 0, chr(952) + '1'],
#     [2, np.pi / 2, 'L1', 0, chr(952) + '2'],
#     [3, 0, 'L2', 0, chr(952) + '3'],
#     [4, 0, 'L3', 0, 0]
# ]
# data2 = [
#     [1, 0, 0, 0, th1],
#     [2, np.pi / 2, L1, 0, th2],
#     [3, 0, L2, 0, th3],
#     [4, 0, L3, 0, 0]
# ]
# data3 = [
#     [1, 0, 0, 0, 30 * np.pi / 180],
#     [2, np.pi / 2, 5, 0, 45 * np.pi / 180],
#     [3, 0, 5, 0, 30 * np.pi / 180],
#     [4, 0, 2.5, 0, 0]
# ]
# xmin = -10
# xmax = 10
# ymin = -5
# ymax = 5
# zmin = -10
# zmax = 10

# # 1
# data = [
#     [1, 0, 0, 'L1+L2', chr(952) + '1'],
#     [2, np.pi / 2, 0, 0, chr(952) + '2'],
#     [3, 0, 'L3', 0, chr(952) + '3'],
#     [4, 0, 'L4', 0, 0],
#     [5, 0, 0, 0, 0],
#     [6, 0, 0, 0, 0]
# ]
# data2 = [
#     [1, 0, 0, L1+L2, th1],
#     [2, np.pi / 2, 0, 0, th2],
#     [3, 0, L3, 0, th3],
#     [4, 0, L4, 0, 0],
#     [5, 0, 0, 0, 0],
#     [6, 0, 0, 0, 0]
# ]
# data3 = [
#     [1, 0, 0, 3+4, 20 * np.pi / 180],
#     [2, np.pi / 2, 0, 0, -20 * np.pi / 180],
#     [3, 0, 8, 0, -20 * np.pi / 180],
#     [4, 0, 4, 0, 0],
#     [5, 0, 0, 0, 0],
#     [6, 0, 0, 0, 0]
# ]
# xmin = -2
# xmax = 15
# ymin = -5
# ymax = 5
# zmin = -2
# zmax = 10

# # 2
# data = [
#     [1, 0, 0, 0, chr(952) + '1'],
#     [2, np.pi / 2, 'a1', 0, 0],
#     [3, 0, 0, 'd2 + L3', np.pi / 2],
#     [4, '-'+chr(952) + '3', 0, 'a4', 0]
# ]
# data2 = [
#     [1, 0, 0, 0, th1],
#     [2, np.pi / 2, a1, 0, 0],
#     [3, 0, 0, d2+L3, np.pi / 2],
#     [4, -th3, 0, a4, 0]
# ]
# data3 = [
#     [1, 0, 0, 0, 30 * np.pi / 180],
#     [2, np.pi / 2, 5, 0, 0],
#     [3, 0, 0, 5+10, np.pi / 2],
#     [4, -45 * np.pi / 180, 0, 4, 0]
# ]
# xmin=-5
# xmax=18
# ymin=-22
# ymax=2
# zmin=-2
# zmax=2

# # 3
# data = [
#     [1, 0, 0, 0, chr(952) + '1'],
#     [2, 0, 'L1', 0, chr(952) + '2'],
#     [3, 0, 'L2', 0, chr(952) + '3'],
#     [4, 0, 'L3', 0, 0]
# ]
# data2 = [
#     [1, 0, 0, 0, th1],
#     [2, 0, L1, 0, th2],
#     [3, 0, L2, 0, th3],
#     [4, 0, L3, 0, 0]
# ]
# data3 = [
#     [1, 0, 0, 0, 30 * np.pi / 180],
#     [2, 0, 10, 0, 40 * np.pi / 180],
#     [3, 0, 10, 0, 70 * np.pi / 180],
#     [4, 0, 3, 0, 0]
# ]
# xmin=-5
# xmax=15
# ymin=-5
# ymax=20
# zmin=-2
# zmax=2

# # 4
# data = [
#     [1, 0, 0, 'a1', chr(952) + '1'],
#     [2, '-'+chr(952) + '2', 0, 'a2', -np.pi / 2],
#     [3, 0, 'd3', 0, 0],
#     [4, 0, 'a3', 0, 0]
# ]
# data2 = [
#     [1, 0, 0, a1, th1],
#     [2, -th2, 0, a2, -np.pi / 2],
#     [3, 0, d3, 0, 0],
#     [4, 0, a3, 0, 0]
# ]
# data3 = [
#     [1, 0, 0, 10, 30 * np.pi / 180],
#     [2, -30 * np.pi / 180, 0, 3, -np.pi / 2],
#     [3, 0, 10, 0, 0],
#     [4, 0, 2, 0, 0]
# ]
# xmin=-5
# xmax=9
# ymin=-10
# ymax=5
# zmin=-2
# zmax=17

# SCARA
data = [
    [1, 0, 0, 'L1', chr(952) + '1'],
    [2, 0, 'L2', 0, 0],
    [3, 0, 0, 'a1', chr(952) + '2'],
    [4, chr(960), 'L3', 0, 0],
    [5, 0, 0, 'd3', 0],
    [6, 0, 0, 'a2', 0]
]
data2 = [
    [1, 0, 0, L1, th1],
    [2, 0, L2, 0, 0],
    [3, 0, 0, a1, th2],
    [4, np.pi, L3, 0, 0],
    [5, 0, 0, d3, 0],
    [6, 0, 0, a2, 0]
]
# data3 = [
#     [1, 0, 0, 75, 45 * np.pi / 180],
#     [2, 0, 40, 0, 0],
#     [3, 0, 0, 15, -70 * np.pi / 180],
#     [4, np.pi, 40, 0, 0],
#     [5, 0, 0, 30, 0],
#     [6, 0, 0, 40, 0]
# ]
data3 = [
    [1, 0, 0, 75, invKin[0][0]],
    [2, 0, 40, 0, 0],
    [3, 0, 0, 15, invKin[0][1]],
    [4, np.pi, 40, 0, 0],
    [5, 0, 0, invKin[0][2], 0],
    [6, 0, 0, 40, 0]
]
xmin=-50
xmax=50
ymin=-50
ymax=60
zmin=-2
zmax=102

print(tabulate(data, headers=['Link', chr(
    945) + 'i-1', 'ai-1', 'di', chr(952) + 'i']))

# Symbolic
a1_s = a1_v.subs([(a10, data[0][1]), (a11, data[1][1]), (a12, data[2][1]), (a13, data[3][1]), (a14, data[4][1]), (a15, data[5][1])])
a_s = a_v.subs([(a1, data[0][2]), (a2, data[1][2]), (a3, data[2][2]), (a4, data[3][2]), (a5, data[4][2]), (a6, data[5][2])])
d_s = d_v.subs([(d1, data[0][3]), (d2, data[1][3]), (d3, data[2][3]), (d4, data[3][3]), (d5, data[4][3]), (d6, data[5][3])])
th_s = th_v.subs([(th1, data[0][4]), (th2, data[1][4]), (th3, data[2][4]), (th4, data[3][4]), (th5, data[4][4]), (th6, data[5][4])])
# Numeric and symbolic
a1_ns = a1_v.subs([(a10, data2[0][1]), (a11, data2[1][1]), (a12, data2[2][1]), (a13, data2[3][1]), (a14, data2[4][1]), (a15, data2[5][1])])
a_ns = a_v.subs([(a1, data2[0][2]), (a2, data2[1][2]), (a3, data2[2][2]), (a4, data2[3][2]), (a5, data2[4][2]), (a6, data2[5][2])])
d_ns = d_v.subs([(d1, data2[0][3]), (d2, data2[1][3]), (d3, data2[2][3]), (d4, data2[3][3]), (d5, data2[4][3]), (d6, data2[5][3])])
th_ns = th_v.subs([(th1, data2[0][4]), (th2, data2[1][4]), (th3, data2[2][4]), (th4, data2[3][4]), (th5, data2[4][4]), (th6, data2[5][4])])

# Numeric for the plot
a1_n = a1_v.subs([(a10, data3[0][1]), (a11, data3[1][1]), (a12, data3[2][1]), (a13, data3[3][1]), (a14, data3[4][1]), (a15, data3[5][1])])
a_n = a_v.subs([(a1, data3[0][2]), (a2, data3[1][2]), (a3, data3[2][2]), (a4, data3[3][2]), (a5, data3[4][2]), (a6, data3[5][2])])
d_n = d_v.subs([(d1, data3[0][3]), (d2, data3[1][3]), (d3, data3[2][3]), (d4, data3[3][3]), (d5, data3[4][3]), (d6, data3[5][3])])
th_n = th_v.subs([(th1, data3[0][4]), (th2, data3[1][4]), (th3, data3[2][4]), (th4, data3[3][4]), (th5, data3[4][4]), (th6, data3[5][4])])

T_o = []
T_s = []
T_ns = []
T = []
for i in range(6):
    T_o.append(sp.Matrix([
        [cos(th_v[i]), -sin(th_v[i]), 0, a_v[i]],
        [sin(th_v[i]) * cos(a1_v[i]), cos(th_v[i]) *
         cos(a1_v[i]), -sin(a1_v[i]), -sin(a1_v[i]) * d_v[i]],
        [sin(th_v[i]) * sin(a1_v[i]), cos(th_v[i]) *
         sin(a1_v[i]), cos(a1_v[i]), cos(a1_v[i]) * d_v[i]],
        [0, 0, 0, 1],
    ])
    )
    T_s.append(sp.Matrix([
        [cos(th_s[i]), -sin(th_s[i]), 0, a_s[i]],
        [sin(th_s[i]) * cos(a1_s[i]), cos(th_s[i]) *
         cos(a1_s[i]), -sin(a1_s[i]), -sin(a1_s[i]) * d_s[i]],
        [sin(th_s[i]) * sin(a1_s[i]), cos(th_s[i]) *
         sin(a1_s[i]), cos(a1_s[i]), cos(a1_s[i]) * d_s[i]],
        [0, 0, 0, 1],
    ])
    )
    T_ns.append(sp.Matrix([
        [cos(th_ns[i]), -sin(th_ns[i]), 0, a_ns[i]],
        [sin(th_ns[i]) * cos(a1_ns[i]), cos(th_ns[i]) *
         cos(a1_ns[i]), -sin(a1_ns[i]), -sin(a1_ns[i]) * d_ns[i]],
        [sin(th_ns[i]) * sin(a1_ns[i]), cos(th_ns[i]) *
         sin(a1_ns[i]), cos(a1_ns[i]), cos(a1_ns[i]) * d_ns[i]],
        [0, 0, 0, 1],
    ])
    )
    T.append(sp.Matrix([
        [cos(th_n[i]), -sin(th_n[i]), 0, a_n[i]],
        [sin(th_n[i]) * cos(a1_n[i]), cos(th_n[i]) *
         cos(a1_n[i]), -sin(a1_n[i]), -sin(a1_n[i]) * d_n[i]],
        [sin(th_n[i]) * sin(a1_n[i]), cos(th_n[i]) *
         sin(a1_n[i]), cos(a1_n[i]), cos(a1_n[i]) * d_n[i]],
        [0, 0, 0, 1],
    ])
    )

print()
print('Matrices:')
print('Denavit-Hartenberg symbolic matrices')
print(T_o)
print('\nSubstituting values from the table on the D-H matrices')
print(T_s)
print('\nSubstituting python symbolic variables on the table')
print(tabulate(data2, headers=['Link', chr(
    945) + 'i-1', 'ai-1', 'di', chr(952) + 'i']))
print()
th_d = [th1,th2,th3,th4,th5,th6,d1,d2,d3,d4,d5,d6]
dof = []
for i in th_d:
    for j in th_ns:
        if i==j:
            dof.append(i)
    for j in d_ns:
        if i==j:
            dof.append(i)
print('Degrees of freedom of the system: ',dof)
print('\nSubstituting python symbolic variables on the matrices')
# for i in range(len(T_ns)):
#     pprint(T_ns[i],use_unicode=False)
print(T_ns)

print(f"\nT01, T12, T23, T34, T45, T56")

a = 0
b = 1
for mat in T_ns:
    print('T' + str(a) + str(b))
    print(mat, '\n')
    a += 1
    b += 1

T01_ns = T_ns[0]
T02_ns = T01_ns * T_ns[1]
T03_ns = T02_ns * T_ns[2]
T04_ns = T03_ns * T_ns[3]
T05_ns = T04_ns * T_ns[4]
T06_ns = T05_ns * T_ns[5]
print('Multiplications')
a = 1
b = 2
T_ns_final_from_0 = [T01_ns,T02_ns,T03_ns,T04_ns,T05_ns,T06_ns]
print('\nT01\n', nsimplify(T01_ns, tolerance=1e-10, rational=True))
for i in range(len(T_ns_final_from_0)):
    if i == 0:
        continue
    print(f'\nT0{b} = T0{a}*T{a}{b}\nT0{b}\n', nsimplify(T_ns_final_from_0[i], tolerance=1e-10, rational=True))
    a += 1
    b += 1
# print('T02\n', nsimplify(T02_ns, tolerance=1e-10, rational=True))
# print('T03\n', nsimplify(T03_ns, tolerance=1e-10, rational=True))
# print('T04\n', nsimplify(T04_ns, tolerance=1e-10, rational=True))
# print('T05\n', nsimplify(T05_ns, tolerance=1e-10, rational=True))
# print('Final position of the end effector with respect to {0}')
# for row in nsimplify(T04_ns, tolerance=1e-10, rational=True):
#     print(sp.factor(row))
#     print()

# print('T06\n', nsimplify(T06_ns, tolerance=1e-10, rational=True))
print('\ndet(T06)\n', T06_ns.det())
print('det(T06)\n', sp.factor(T06_ns.det()))


print(
    f"\nNow Substituting numeric values from {chr(945)}, a, d and {chr(952)} values in T01, T12, T23, T34, T45, T56")

a = 0
b = 1
for mat in T:
    # print('T' + str(a) + str(b))
    # print(mat, '\n')
    cpprint('\nT' + str(a) + str(b),mat)
    a += 1
    b += 1

# This is the last translation from the table (a or d), therefore d6
P5H = sp.Matrix([[0], [0], [40], [1]])
T01 = T[0]
T02 = T01 * T[1]
T03 = T02 * T[2]
T04 = T03 * T[3]
T05 = T04 * T[4]
T06 = T05 * T[5]
P0H = T05 * P5H
print('\nMultiplications')
a = 1
b = 2
T_final_from_0 = [T01,T02,T03,T04,T05,T06]
cpprint('\nT01\n', T01)
for i in range(len(T_final_from_0)):
    if i == 0:
        continue
    cpprint(f'\nT0{b} = T0{a}*T{a}{b}\nT0{b}', T_final_from_0[i])
    a += 1
    b += 1
# cpprint('\nT03 = T02*T23\n', T03)
# cpprint('\nT04 = T03*T34\n', T04)
# cpprint('\nT05 = T04*T45\n', T05)
# cpprint('\nT06 = T05*T56\n', T06)
cpprint('\nP0H', P0H)  # this must be equals to the last column of T06

# Define unit vectors
i0 = sp.Matrix([1, 0, 0, 1])
j0 = sp.Matrix([0, 1, 0, 1])
k0 = sp.Matrix([0, 0, 1, 1])

# print()
# print('Homogeneous Transformations shape')
# print(T01.shape)
# print('Unit vectors shape')
# print(i0.shape)

i01 = T01 * i0
j01 = T01 * j0
k01 = T01 * k0

i02 = T02 * i0
j02 = T02 * j0
k02 = T02 * k0

i03 = T03 * i0
j03 = T03 * j0
k03 = T03 * k0

i04 = T04 * i0
j04 = T04 * j0
k04 = T04 * k0

i05 = T05 * i0
j05 = T05 * j0
k05 = T05 * k0

i06 = T06 * i0
j06 = T06 * j0
k06 = T06 * k0

size = 7
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.quiver(0, 0, 0, size*(i0[0]), size*(i0[1]), size*(i0[2]), color='red', linewidth=3)
ax.quiver(0, 0, 0, size*(j0[0]), size*(j0[1]), size*(j0[2]), color='green', linewidth=3)
ax.quiver(0, 0, 0, size*(k0[0]), size*(k0[1]), size*(k0[2]), color='blue', linewidth=3)

# [1]
ax.plot3D(np.linspace(float(0), float(T01[0, 3])), np.linspace(float(0), float(T01[1, 3])), np.linspace(float(0), float(T01[2, 3])), linewidth=3)
ax.quiver(T01[0,3], T01[1,3], T01[2,3], size*(i01[0] - T01[0,3]), size*(i01[1] - T01[1,3]), size*(i01[2] - T01[2,3]), color='red', linewidth=3)
ax.quiver(T01[0,3], T01[1,3], T01[2,3], size*(j01[0] - T01[0,3]), size*(j01[1] - T01[1,3]), size*(j01[2] - T01[2,3]), color='green', linewidth=3)
ax.quiver(T01[0,3], T01[1,3], T01[2,3], size*(k01[0] - T01[0,3]), size*(k01[1] - T01[1,3]), size*(k01[2] - T01[2,3]), color='blue', linewidth=3)

# [2]
ax.plot3D(np.linspace(float(T01[0,3]), float(T02[0,3])), np.linspace(float(T01[1,3]), float(T02[1,3])), np.linspace(float(T01[2,3]), float(T02[2,3])), linewidth=3)
ax.quiver(T02[0,3], T02[1,3], T02[2,3], size*(i02[0] - T02[0,3]), size*(i02[1] - T02[1,3]), size*(i02[2] - T02[2,3]), color='red', linewidth=3)
ax.quiver(T02[0,3], T02[1,3], T02[2,3], size*(j02[0] - T02[0,3]), size*(j02[1] - T02[1,3]), size*(j02[2] - T02[2,3]), color='green', linewidth=3)
ax.quiver(T02[0,3], T02[1,3], T02[2,3], size*(k02[0] - T02[0,3]), size*(k02[1] - T02[1,3]), size*(k02[2] - T02[2,3]), color='blue', linewidth=3)
# [3]
ax.plot3D(np.linspace(float(T02[0,3]), float(T03[0,3])), np.linspace(float(T02[1,3]), float(T03[1,3])), np.linspace(float(T02[2,3]), float(T03[2,3])), linewidth=3)
ax.quiver(T03[0,3], T03[1,3], T03[2,3], size*(i03[0] - T03[0,3]), size*(i03[1] - T03[1,3]), size*(i03[2] - T03[2,3]), color='red', linewidth=3)
ax.quiver(T03[0,3], T03[1,3], T03[2,3], size*(j03[0] - T03[0,3]), size*(j03[1] - T03[1,3]), size*(j03[2] - T03[2,3]), color='green', linewidth=3)
ax.quiver(T03[0,3], T03[1,3], T03[2,3], size*(k03[0] - T03[0,3]), size*(k03[1] - T03[1,3]), size*(k03[2] - T03[2,3]), color='blue', linewidth=3)
# [4]
ax.plot3D(np.linspace(float(T03[0,3]), float(T04[0,3])), np.linspace(float(T03[1,3]), float(T04[1,3])), np.linspace(float(T03[2,3]), float(T04[2,3])), linewidth=3)
ax.quiver(T04[0,3], T04[1,3], T04[2,3], size*(i04[0] - T04[0,3]), size*(i04[1] - T04[1,3]), size*(i04[2] - T04[2,3]), color='red', linewidth=3)
ax.quiver(T04[0,3], T04[1,3], T04[2,3], size*(j04[0] - T04[0,3]), size*(j04[1] - T04[1,3]), size*(j04[2] - T04[2,3]), color='green', linewidth=3)
ax.quiver(T04[0,3], T04[1,3], T04[2,3], size*(k04[0] - T04[0,3]), size*(k04[1] - T04[1,3]), size*(k04[2] - T04[2,3]), color='blue', linewidth=3)
# [5]
ax.plot3D(np.linspace(float(T04[0,3]), float(T05[0,3])), np.linspace(float(T04[1,3]), float(T05[1,3])), np.linspace(float(T04[2,3]), float(T05[2,3])), linewidth=3)
ax.quiver(T05[0,3], T05[1,3], T05[2,3], size*(i05[0] - T05[0,3]), size*(i05[1] - T05[1,3]), size*(i05[2] - T05[2,3]), color='red', linewidth=3)
ax.quiver(T05[0,3], T05[1,3], T05[2,3], size*(j05[0] - T05[0,3]), size*(j05[1] - T05[1,3]), size*(j05[2] - T05[2,3]), color='green', linewidth=3)
ax.quiver(T05[0,3], T05[1,3], T05[2,3], size*(k05[0] - T05[0,3]), size*(k05[1] - T05[1,3]), size*(k05[2] - T05[2,3]), color='blue', linewidth=3)
# [6]
ax.plot3D(np.linspace(float(T05[0,3]), float(T06[0,3])), np.linspace(float(T05[1,3]), float(T06[1,3])), np.linspace(float(T05[2,3]), float(T06[2,3])), linewidth=3)
ax.quiver(T06[0,3], T06[1,3], T06[2,3], size*(i06[0] - T06[0,3]), size*(i06[1] - T06[1,3]), size*(i06[2] - T06[2,3]), color='red', linewidth=3)
ax.quiver(T06[0,3], T06[1,3], T06[2,3], size*(j06[0] - T06[0,3]), size*(j06[1] - T06[1,3]), size*(j06[2] - T06[2,3]), color='green', linewidth=3)
ax.quiver(T06[0,3], T06[1,3], T06[2,3], size*(k06[0] - T06[0,3]), size*(k06[1] - T06[1,3]), size*(k06[2] - T06[2,3]), color='blue', linewidth=3)
ax.set_title('3D Plot SCARA')
ax.set_xlabel('Eje X_0')
ax.set_ylabel('Eje Y_0')
ax.set_zlabel('Eje Z_0')
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_zlim(zmin, zmax)
plt.show()


# Unit vectors
x_u = sp.Matrix([1, 0, 0])
z_u = sp.Matrix([0, 0, 1])
print("\nVelocity")
# Derivatives
# Rotational terms
print(f'column {chr(952)}+i')
print('i=1 = ',th_ns[0])
th1p = [th1_p if sp.diff(th_ns[0], th1)!=0 or sp.diff(a1_ns[0], th1)!=0 else 0]
print('th1p = ',th1p)
print('i=2 = ',th_ns[1])
th2p = [th2_p if sp.diff(th_ns[1], th2)!=0 or sp.diff(a1_ns[1], th2)!=0 else 0]
th2p = [th1_p if sp.diff(th_ns[1], th1)!=0 or sp.diff(a1_ns[1], th1)!=0 else th2p[0]]
print('th2p = ',th2p)
print('i=3 = ',th_ns[2])
th3p = [th3_p if sp.diff(th_ns[2], th3)!=0 or sp.diff(a1_ns[2], th3)!=0 else 0]
th3p = [th2_p if sp.diff(th_ns[2], th2)!=0 or sp.diff(a1_ns[2], th2)!=0 else th3p[0]]
th3p = [th1_p if sp.diff(th_ns[2], th1)!=0 or sp.diff(a1_ns[2], th1)!=0 else th3p[0]]
print('th3p = ',th3p)
print('i=4 = ',th_ns[3])
th4p = [th4_p if sp.diff(th_ns[3], th4)!=0 or sp.diff(a1_ns[3], th4)!=0 else 0]
th4p = [th3_p if sp.diff(th_ns[3], th3)!=0 or sp.diff(a1_ns[3], th3)!=0  else th4p[0]]
th4p = [th2_p if sp.diff(th_ns[3], th2)!=0 or sp.diff(a1_ns[3], th2)!=0  else th4p[0]]
print('th4p = ',th4p)
print('i=5 = ',th_ns[4])
th5p = [th5_p if sp.diff(th_ns[4], th5)!=0 or sp.diff(a1_ns[4], th5)!=0 else 0]
th5p = [th4_p if sp.diff(th_ns[4], th4)!=0 or sp.diff(a1_ns[4], th4)!=0  else th5p[0]]
th5p = [th3_p if sp.diff(th_ns[4], th3)!=0 or sp.diff(a1_ns[4], th3)!=0  else th5p[0]]
print('th5p = ',th5p)
print('i=6 = ',th_ns[5])
th6p = [th6_p if sp.diff(th_ns[5], th6)!=0 or sp.diff(a1_ns[5], th6)!=0 else 0]
th6p = [th5_p if sp.diff(th_ns[5], th5)!=0 or sp.diff(a1_ns[5], th5)!=0  else th6p[0]]
th6p = [th4_p if sp.diff(th_ns[5], th4)!=0 or sp.diff(a1_ns[5], th4)!=0  else th6p[0]]
print('th6p = ',th6p)
# Translation terms
print('column di')
print('i=1 = ',d_ns[0])
d1p = [d1_p if sp.diff(d_ns[0], d1)!=0 or sp.diff(a_ns[0], d1)!=0 else 0]
print('d1p = ',d1p)
print('i=2 = ',d_ns[1])
d2p = [d2_p if sp.diff(d_ns[1], d2)!=0 or sp.diff(a_ns[1], d2)!=0 else 0]
d2p = [d1_p if sp.diff(d_ns[1], d1)!=0 or sp.diff(a_ns[1], d1)!=0 else d2p[0]]
print('d2p = ',d2p)
print('i=3 = ',d_ns[2])
d3p = [d3_p if sp.diff(d_ns[2], d3)!=0 or sp.diff(a_ns[2], d3)!=0 else 0]
d3p = [d2_p if sp.diff(d_ns[2], d2)!=0 or sp.diff(a_ns[2], d2)!=0 else d3p[0]]
d3p = [d1_p if sp.diff(d_ns[2], d1)!=0 or sp.diff(a_ns[2], d1)!=0 else d3p[0]]
print('d3p = ',d3p)
print('i=4 = ',d_ns[3])
d4p = [d4_p if sp.diff(d_ns[3], d4)!=0 or sp.diff(a_ns[3], d4)!=0 else 0]
d4p = [d3_p if sp.diff(d_ns[3], d3)!=0 or sp.diff(a_ns[3], d3)!=0 else d4p[0]]
d4p = [d2_p if sp.diff(d_ns[3], d2)!=0 or sp.diff(a_ns[3], d2)!=0 else d4p[0]]
print('d4p = ',d4p)
print('i=5 = ',d_ns[4])
d5p = [d5_p if sp.diff(d_ns[4], d5)!=0 or sp.diff(a_ns[4], d5)!=0 else 0]
d5p = [d4_p if sp.diff(d_ns[4], d4)!=0 or sp.diff(a_ns[4], d4)!=0 else d5p[0]]
d5p = [d3_p if sp.diff(d_ns[4], d3)!=0 or sp.diff(a_ns[4], d3)!=0 else d5p[0]]
print('d5p = ',d5p)
print('i=6 = ',d_ns[5])
d6p = [d6_p if sp.diff(d_ns[5], d6)!=0 or sp.diff(a_ns[5], d6)!=0 else 0]
d6p = [d5_p if sp.diff(d_ns[5], d5)!=0 or sp.diff(a_ns[5], d5)!=0 else d6p[0]]
d6p = [d4_p if sp.diff(d_ns[5], d4)!=0 or sp.diff(a_ns[5], d4)!=0 else d6p[0]]
print('d6p = ',d6p)

th_d_p = [th1p,th2p,th3p,th4p,th5p,th6p,d1p,d2p,d3p,d4p,d5p,d6p]
dof_p = [i[0] for i in th_d_p if i[0]!=0]
print('\nDerivatives of the degrees of freedom of the system:',dof_p)
print()
# Linear & angular initial velocities
w00 = sp.Matrix([0, 0, 0])
v00 = sp.Matrix([0, 0, 0])

# Rotationals
print("Rotational matrices:")
R01 = T_ns[0][:3, :3]
R12 = nsimplify(T_ns[1][:3, :3], tolerance=1e-10, rational=True)
R23 = T_ns[2][:3, :3]
R34 = nsimplify(T_ns[3][:3, :3], tolerance=1e-10, rational=True)
R45 = T_ns[4][:3, :3]
R56 = T_ns[5][:3, :3]
print('R01: ', R01)
print('R12: ', R12)
print('R23: ', R23)
print('R34: ', R34)
print('R45: ', R45)
print('R56: ', R56)

print("Inverse of previous rotational matrices:")
R10 = R01.T
R21 = R12.T
R32 = R23.T
R43 = R34.T
R54 = R45.T
R65 = R56.T
print('R10: ', R10)
print('R21: ', R21)
print('R32: ', R32)
print('R43: ', R43)
print('R54: ', R54)
print('R65: ', R65)

# Position vectors
print('Position Vectors')
P01 = T_ns[0][:3, 3]
P12 = T_ns[1][:3, 3]
P23 = T_ns[2][:3, 3]
P34 = T_ns[3][:3, 3]
P45 = T_ns[4][:3, 3]
P56 = T_ns[5][:3, 3]
print('P01: ', P01)
print('P12: ', P12)
print('P23: ', P23)
print('P34: ', P34)
print('P45: ', P45)
print('P56: ', P56)

# Operations
print('\nOperations\n')
print(f'w11 = R10*w00 + {chr(952)}1p*Z1_u: ')
w11 = R10 * w00 + th1p[0] * z_u
print('w11: ', w11)
print('v11 = R10(v00 + w00 x P01) + d1p*Z1_u: ')
v11 = R10 * (v00 + w00.cross(P01)) + d1p[0] * z_u
print('v11: ', v11)
print()

print(f'w22 = R21*w11 + {chr(952)}2p*Z2_u: ')
w22 = R21 * w11 + th2p[0] * z_u
print('w22: ', w22)
w22_xyz = sp.Matrix([om22x, om22y, om22z])
print('v22 = R21(v11 + w11 x P12) + d2p*Z2_u: ')
v22 = R21 * (v11 + w11.cross(P12)) + d2p[0] * z_u
print('v22: ', v22)
print()

print(f'w33 = R32*w22 + {chr(952)}3p*Z3_u: ')
w33 = R32 * w22 + th3p[0] * z_u
print('w33: ', w33)
w33_xyz = sp.Matrix([om33x, om33y, om33z])
print('v33 = R32(v22 + w22 x P23) + d3p*Z3_u: ')
v33 = R32 * (v22 + w22.cross(P23)) + d3p[0] * z_u
print('v33: ', v33)
print()
# R03 = nsimplify(T03_ns[:3, :3], tolerance=1e-10, rational=True)
# print("We need another Rotational matrix")
# print('R03: ', R03)
# v03 = R03 * v33
# print('v03 = R03*v33: ')
# print('v03: ', v03)
# print()
print(f'w44 = R43*w33 +{chr(952)}4p*Z4_u: ')
w44 = R43 * w33 + th4p[0] * z_u
print('w44: ', w44)
w44_xyz = sp.Matrix([om44x, om44y, om44z])
print('v44 = R43(v33 + w33 x P34) + d4p*Z4_u: ')
v44 = R43 * (v33 + w33.cross(P34)) + d4p[0] * z_u
print('v44: ', v44)
print()

print(f'w55 = R54*w44 +{chr(952)}5p*Z5_u: ')
w55 = R54 * w44 + th5p[0] * z_u
print('w55: ', w55)
w55_xyz = sp.Matrix([om55x, om55y, om55z])
print('v55 = R54(v44 + w44 x P45) + d5p*Z5_u: ')
v55 = R54 * (v44 + w44.cross(P45)) + d5p[0] * z_u
print('v55: ', v55)
print()

print(f'w66 = R65*w55 +{chr(952)}6p*Z6_u: ')
w66 = R65 * w55 + th6p[0] * z_u
print('w66: ', w66)
w66_xyz = sp.Matrix([om66x, om66y, om66z])
print('v66 = R65(v55 + w55 x P56) + d6p*Z6_u: ')
v66 = R65 * (v55 + w55.cross(P56)) + d6p[0] * z_u
print('v66: ', v66)
print()
R06 = nsimplify(T06_ns[:3, :3], tolerance=1e-10, rational=True)

print("Calculate point velocity from origin")
print('R06: ', R06)
v06 = R06 * v66
print('v06 = R06*v66: ')
print('v06: ', v06)
print()

# Acceleration
print("Acceleration")
# Derivatives
# Rotational terms
print(f'column {chr(952)}+i_p')
print('i=1 = ',th_ns[0])
th1pp = [th1_pp if sp.diff(th_ns[0], th1)!=0 or sp.diff(a1_ns[0], th1)!=0 else 0]
print('th1pp = ',th1pp)
print('i=2 = ',th_ns[1])
th2pp = [th2_pp if sp.diff(th_ns[1], th2)!=0 or sp.diff(a1_ns[1], th2)!=0 else 0]
th2pp = [th1_pp if sp.diff(th_ns[1], th1)!=0 or sp.diff(a1_ns[1], th1)!=0 else th2pp[0]]
print('th2pp = ',th2pp)
print('i=3 = ',th_ns[2])
th3pp = [th3_pp if sp.diff(th_ns[2], th3)!=0 or sp.diff(a1_ns[2], th3)!=0 else 0]
th3pp = [th2_pp if sp.diff(th_ns[2], th2)!=0 or sp.diff(a1_ns[2], th2)!=0 else th3pp[0]]
th3pp = [th1_pp if sp.diff(th_ns[2], th1)!=0 or sp.diff(a1_ns[2], th1)!=0 else th3pp[0]]
print('th3pp = ',th3pp)
print('i=4 = ',th_ns[3])
th4pp = [th4_pp if sp.diff(th_ns[3], th4)!=0 or sp.diff(a1_ns[3], th4)!=0 else 0]
th4pp = [th3_pp if sp.diff(th_ns[3], th3)!=0 or sp.diff(a1_ns[3], th3)!=0  else th4pp[0]]
th4pp = [th2_pp if sp.diff(th_ns[3], th2)!=0 or sp.diff(a1_ns[3], th2)!=0  else th4pp[0]]
print('th4pp = ',th4pp)
print('i=5 = ',th_ns[4])
th5pp = [th5_pp if sp.diff(th_ns[4], th5)!=0 or sp.diff(a1_ns[4], th5)!=0 else 0]
th5pp = [th4_pp if sp.diff(th_ns[4], th4)!=0 or sp.diff(a1_ns[4], th4)!=0  else th5pp[0]]
th5pp = [th3_pp if sp.diff(th_ns[4], th3)!=0 or sp.diff(a1_ns[4], th3)!=0  else th5pp[0]]
print('th5pp = ',th5pp)
print('i=6 = ',th_ns[5])
th6pp = [th6_pp if sp.diff(th_ns[5], th6)!=0 or sp.diff(a1_ns[5], th6)!=0 else 0]
th6pp = [th5_pp if sp.diff(th_ns[5], th5)!=0 or sp.diff(a1_ns[5], th5)!=0  else th6pp[0]]
th6pp = [th4_pp if sp.diff(th_ns[5], th4)!=0 or sp.diff(a1_ns[5], th4)!=0  else th6pp[0]]
print('th6pp = ',th6pp)
# Translation terms
print('column di_p')
print('i=1 = ',d_ns[0])
d1pp = [d1_pp if sp.diff(d_ns[0], d1)!=0 or sp.diff(a_ns[0], d1)!=0 else 0]
print('d1pp = ',d1pp)
print('i=2 = ',d_ns[1])
d2pp = [d2_pp if sp.diff(d_ns[1], d2)!=0 or sp.diff(a_ns[1], d2)!=0 else 0]
d2pp = [d1_pp if sp.diff(d_ns[1], d1)!=0 or sp.diff(a_ns[1], d1)!=0 else d2pp[0]]
print('d2pp = ',d2pp)
print('i=3 = ',d_ns[2])
d3pp = [d3_pp if sp.diff(d_ns[2], d3)!=0 or sp.diff(a_ns[2], d3)!=0 else 0]
d3pp = [d2_pp if sp.diff(d_ns[2], d2)!=0 or sp.diff(a_ns[2], d2)!=0 else d3pp[0]]
d3pp = [d1_pp if sp.diff(d_ns[2], d1)!=0 or sp.diff(a_ns[2], d1)!=0 else d3pp[0]]
print('d3pp = ',d3pp)
print('i=4 = ',d_ns[3])
d4pp = [d4_pp if sp.diff(d_ns[3], d4)!=0 or sp.diff(a_ns[3], d4)!=0 else 0]
d4pp = [d3_pp if sp.diff(d_ns[3], d3)!=0 or sp.diff(a_ns[3], d3)!=0 else d4pp[0]]
d4pp = [d2_pp if sp.diff(d_ns[4], d2)!=0 or sp.diff(a_ns[4], d2)!=0 else d4pp[0]]
print('d4pp = ',d4pp)
print('i=5 = ',d_ns[4])
d5pp = [d5_pp if sp.diff(d_ns[4], d5)!=0 or sp.diff(a_ns[4], d5)!=0 else 0]
d5pp = [d4_pp if sp.diff(d_ns[4], d4)!=0 or sp.diff(a_ns[4], d4)!=0 else d5pp[0]]
d5pp = [d3_pp if sp.diff(d_ns[4], d3)!=0 or sp.diff(a_ns[4], d3)!=0 else d5pp[0]]
print('d5pp = ',d5pp)
print('i=6 = ',d_ns[5])
d6pp = [d6_pp if sp.diff(d_ns[5], d6)!=0 or sp.diff(a_ns[5], d6)!=0 else 0]
d6pp = [d5_pp if sp.diff(d_ns[5], d5)!=0 or sp.diff(a_ns[5], d5)!=0 else d6pp[0]]
d6pp = [d4_pp if sp.diff(d_ns[5], d4)!=0 or sp.diff(a_ns[5], d4)!=0 else d6pp[0]]
print('d6pp = ',d6pp)

# Linear & angular initial accelerations
w00p = sp.Matrix([0, 0, 0])
v00p = sp.Matrix([0, 0, 0])

# Operations
print('\nOperations\n')
print(f'w11p = R10*w00p + R10*w00 x {chr(952)}1p*Z1_u + {chr(952)}1pp*Z1_u: ')
w11p = R10 * w00p + (R10 * w00).cross(th1p[0] * z_u) + th1pp[0] * z_u
print('w11p: ', w11p)
print('v11p = R10(v00p + w00p x P01 + w00 x (w00 x P01)) + 2*(w11 x d1p*Z1_u) + d1pp*Z1_u: ')
v11p = R10 * (v00p + w00p.cross(P01) + w00.cross(w00.cross(P01))) + 2*(w11.cross(d1p[0] * z_u)) + d1pp[0] * z_u
print('v11p: ', v11p)
print()

print(f'w22p = R21*w11p + R21*w11 x {chr(952)}2p*Z2_u + {chr(952)}2pp*Z2_u: ')
w22p = R21 * w11p + (R21 * w11).cross(th2p[0] * z_u) + th2pp[0] * z_u
print('w22p: ', w22p)
print('v22p = R21(v11p + w11p x P12 + w11 x (w11 x P12)) + 2*(w22 x d2p*Z2_u) + d2pp*Z2_u: ')
v22p = R21 * (v11p + w11p.cross(P12) + w11.cross(w11.cross(P12))) + 2*(w22.cross(d2p[0] * z_u)) + d2pp[0] * z_u
print('v22p: ', v22p)
print()

print(f'w33p = R32*w22p + R32*w22 x {chr(952)}3p*Z3_u + {chr(952)}3pp*Z3_u: ')
w33p = R32 * w22p + (R32 * w22).cross(th3p[0] * z_u) + th3pp[0] * z_u
print('w33p: ', w33p)
print('v33p = R32(v22p + w22p x P23 + w22 x (w22 x P23)) + 2*(w33 x d3p*Z3_u) + d3pp*Z3_u: ')
v33p = R32 * (v22p + w22p.cross(P23) + w22.cross(w22.cross(P23))) + 2*(w33.cross(d3p[0] * z_u)) + d3pp[0] * z_u
print('v33p: ', v33p)
print()
# # Calculate point velocity from origin
# R03 = nsimplify(T03_ns[:3, :3], tolerance=1e-10, rational=True)
# print("We need another Rotational matrix")
# print('R03: ', R03)
# v03 = R03 * v33
# print('v03 = R03*v33: ')
# print('v03: ', v03)
# print()
# Keep going with matrices operations
print(f'w44p = R43*w33p + R43*w33 x {chr(952)}4p*Z4_u + {chr(952)}4pp*Z4_u: ')
w44p = R43 * w33p + (R43 * w33).cross(th4p[0] * z_u) + th4pp[0] * z_u
print('w44p: ', w44p)
print('v44p = R43(v33p + w33p x P34 + w33 x (w33 x P34)) + 2*(w44 x d4p*Z4_u) + d4pp*Z4_u: ')
v44p = R43 * (v33p + w33p.cross(P34) + w33.cross(w33.cross(P34))) + 2*(w44.cross(d4p[0] * z_u)) + d4pp[0] * z_u
print('v44p: ', v44p)
print()

print(f'w55p = R54*w44p + R54*w44 x {chr(952)}5p*Z5_u + {chr(952)}5pp*Z5_u: ')
w55p = R54 * w44p + (R54 * w44).cross(th5p[0] * z_u) + th5pp[0] * z_u
print('w55p: ', w55p)
print('v55p = R54(v44p + w44p x P45 + w44 x (w44 x P45)) + 2*(w55 x d5p*Z5_u) + d5pp*Z5_u: ')
v55p = R54 * (v44p + w44p.cross(P45) + w44.cross(w44.cross(P45))) + 2*(w55.cross(d5p[0] * z_u)) + d5pp[0] * z_u
print('v55p: ', v55p)
print()

print(f'w66p = R65*w55p + R65*w55 x {chr(952)}6p*Z6_u + {chr(952)}6pp*Z6_u: ')
w66p = R65 * w55p + (R65 * w55).cross(th6p[0] * z_u) + th6pp[0] * z_u
print('w66p: ', w66p)
print('v66p = R65(v55p + w55p x P56 + w55 x (w55 x P56)) + 2*(w66 x d6p*Z6_u) + d6pp*Z6_u: ')
v66p = R65 * (v55p + w55p.cross(P56) + w55.cross(w55.cross(P56))) + 2*(w66.cross(d6p[0] * z_u)) + d6pp[0] * z_u
print('v66p: ', v66p)
print()

print("Calculate point acceleration from origin")
print('R06: ', R06)
v06p = R06 * v66p
print('v06p = R06*v66p: ')
print('v06p: ', v06p)
print()

# Taking gravity in consideration
print('Acceleration taking gravity in consideration')
vg = sp.Matrix([0, 0, g])
print('vp = ', vg)
# Center of mass
# P1c1 = sp.Matrix([0, 0, Lc1])
# P2c2 = sp.Matrix([Lc2, 0, 0])
# P3c3 = sp.Matrix([0, 0, Lc3])
# P4c4 = sp.Matrix([Lc4, 0, 0])
# P5c5 = sp.Matrix([0, 0, Lc5])
# P6c6 = sp.Matrix([0, 0, Lc6])
print('Center of mass: ', P1c1, P2c2, P3c3, P4c4, P5c5, P6c6)

# Operations
print('Operations')
print(f'w11p = R10*w00p + R10*w00 x {chr(952)}1p*Z1_u + {chr(952)}1pp*Z1_u: ')
w11p = R10 * w00p + (R10 * w00).cross(th1p[0] * z_u) + th1pp[0] * z_u
print('w11p: ', w11p)
print('v11p = R10(v00p + w00p x P1c1 + w00 x (w00 x P1c1)) + 2*(w11 x d1p*Z1_u) + d1pp*Z1_u: ')
v11p = R10 * (vg + w00p.cross(P1c1) + w00.cross(w00.cross(P1c1))) + 2*(w11.cross(d1p[0] * z_u)) + d1pp[0] * z_u
print('v11p: ', v11p)
print('vc11p = v11p + w11p x P1c1 + w11 x (w11 x P1c1)')
vc11p = v11p + w11p.cross(P1c1) + w11.cross(w11.cross(P1c1))
print('vc11p: ', vc11p)
# pprint(vc11p, use_unicode=False)
print()

print(f'w22p = R21*w11p + R21*w11 x {chr(952)}2p*Z2_u + {chr(952)}2pp*Z2_u: ')
w22p = R21 * w11p + (R21 * w11).cross(th2p[0] * z_u) + th2pp[0] * z_u
print('w22p: ', w22p)
w22p_xyz = sp.Matrix([om22x_p, om22y_p, om22z_p])
print('v22p = R21(v11p + w11p x P12 + w11 x (w11 x P12)) + 2*(w22 x d2p*Z2_u) + d2pp*Z2_u: ')
v22p = R21 * (v11p + w11p.cross(P12) + w11.cross(w11.cross(P12))) + 2*(w22.cross(d2p[0] * z_u)) + d2pp[0] * z_u
print('v22p: ', v22p)
v22p_xyz = sp.Matrix([vel22x_p, vel22y_p, vel22z_p])
print('vc22p = v22p + w22p x P2c2 + w22 x (w22 x P2c2)')
vc22p = v22p + w22p.cross(P2c2) + w22.cross(w22.cross(P2c2))
print('vc22p: ', vc22p)
# pprint(vc22p, use_unicode=False)
print()

print(f'w33p = R32*w22p + R32*w22 x {chr(952)}3p*Z3_u + {chr(952)}3pp*Z3_u: ')
w33p = R32 * w22p + (R32 * w22).cross(th3p[0] * z_u) + th3pp[0] * z_u
w33p_simpl = R32 * w22p_xyz + (R32 * w22_xyz).cross(th3p[0] * z_u) + th3pp[0] * z_u
print('w33p: ', w33p)
print('w33p_simpl: ', w33p_simpl)
w33p_xyz = sp.Matrix([om33x_p, om33y_p, om33z_p])
print('v33p = R32(v22p + w22p x P23 + w22 x (w22 x P23)) + 2*(w33 x d3p*Z3_u) + d3pp*Z3_u: ')
v33p = R32 * (v22p + w22p.cross(P23) + w22.cross(w22.cross(P23))) + 2*(w33.cross(d3p[0] * z_u)) + d3pp[0] * z_u
v33p_simpl = R32 * (v22p_xyz + w22p_xyz.cross(P23) + w22_xyz.cross(w22_xyz.cross(P23))) + 2*(w33_xyz.cross(d3p[0] * z_u)) + d3pp[0] * z_u
print('v33p: ', v33p)
print('v33p_simpl: ', v33p_simpl)
v33p_xyz = sp.Matrix([vel33x_p, vel33y_p, vel33z_p])
print('vc33p = v33p + w33p x P3c3 + w33 x (w33 x P3c3)')
vc33p = v33p + w33p.cross(P3c3) + w33.cross(w33.cross(P3c3))
vc33p_simpl = v33p_xyz + w33p.cross(P3c3) + w33.cross(w33.cross(P3c3))
print('vc33p: ', vc33p)
# pprint(vc33p, use_unicode=False)
print('vc33p_simpl: ', vc33p_simpl)
print()

print(f'w44p = R43*w33p + R43*w33 x {chr(952)}4p*Z4_u + {chr(952)}4pp*Z4_u: ')
w44p = R43 * w33p + (R43 * w33).cross(th4p[0] * z_u) + th4pp[0] * z_u
w44p_simpl = R43 * w33p_xyz + (R43 * w33_xyz).cross(th4p[0] * z_u) + th4pp[0] * z_u
print('w44p: ', w44p)
print('w44p_simpl: ', w44p_simpl)
w44p_xyz = sp.Matrix([om44x_p, om44y_p, om44z_p])
print('v44p = R43(v33p + w33p x P34 + w33 x (w33 x P34)) + 2*(w44 x d4p*Z4_u) + d4pp*Z4_u: ')
v44p = R43 * (v33p + w33p.cross(P34) + w33.cross(w33.cross(P34))) + 2*(w44.cross(d4p[0] * z_u)) + d4pp[0] * z_u
v44p_simpl = R43 * (v33p_xyz + w33p_xyz.cross(P34) + w33_xyz.cross(w33_xyz.cross(P34))) + 2*(w44_xyz.cross(d4p[0] * z_u)) + d4pp[0] * z_u
print('v44p: ', v44p)
print('v44p_simpl: ', v44p_simpl)
v44p_xyz = sp.Matrix([vel44x_p, vel44y_p, vel44z_p])
print('vc44p = v44p + w44p x P4c4 + w44 x (w44 x P4c4)')
vc44p = v44p + w44p.cross(P4c4) + w44.cross(w44.cross(P4c4))
vc44p_simpl = v44p_xyz + w44p_xyz.cross(P4c4) + w44_xyz.cross(w44.cross(P4c4))
print('vc44p: ', vc44p)
# pprint(vc44p, use_unicode=False)
print('vc44p_simpl: ', vc44p_simpl)
print()

print(f'w55p = R54*w44p + R54*w44 x {chr(952)}5p*Z5_u + {chr(952)}5pp*Z5_u: ')
w55p = R54 * w44p + (R54 * w44).cross(th5p[0] * z_u) + th5pp[0] * z_u
w55p_simpl = R54 * w44p_xyz + (R54 * w44_xyz).cross(th5p[0] * z_u) + th5pp[0] * z_u
print('w55p: ', w55p)
print('w55p_simpl: ', w55p_simpl)
w55p_xyz = sp.Matrix([om55x_p, om55y_p, om55z_p])
print('v55p = R54(v44p + w44p x P45 + w44 x (w44 x P45)) + 2*(w55 x d5p*Z5_u) + d5pp*Z5_u: ')
v55p = R54 * (v44p + w44p.cross(P45) + w44.cross(w44.cross(P45))) + 2*(w55.cross(d5p[0] * z_u)) + d5pp[0] * z_u
v55p_simpl = R54 * (v44p_xyz + w44p_xyz.cross(P45) + w44_xyz.cross(w44_xyz.cross(P45))) + 2*(w55_xyz.cross(d5p[0] * z_u)) + d5pp[0] * z_u
print('v55p: ', v55p)
print('v55p_simpl: ', v55p_simpl)
v55p_xyz = sp.Matrix([vel55x_p, vel55y_p, vel55z_p])
print('vc55p = v55p + w55p x P5c5 + w55 x (w55 x P5c5)')
vc55p = v55p + w55p.cross(P5c5) + w55.cross(w55.cross(P5c5))
vc55p_simpl = v55p_xyz + w55p_xyz.cross(P5c5) + w55_xyz.cross(w55.cross(P5c5))
print('vc55p: ', vc55p)
# pprint(vc55p, use_unicode=False)
print('vc55p_simpl: ', vc55p_simpl)
print()

print(f'w66p = R65*w55p + R65*w55 x {chr(952)}6p*Z6_u + {chr(952)}6pp*Z6_u: ')
w66p = R65 * w55p + (R65 * w55).cross(th6p[0] * z_u) + th6pp[0] * z_u
w66p_simpl = R65 * w55p_xyz + (R65 * w55_xyz).cross(th6p[0] * z_u) + th6pp[0] * z_u
print('w66p: ', w66p)
print('w66p_simpl: ', w66p_simpl)
w66p_xyz = sp.Matrix([om66x_p, om66y_p, om66z_p])
print('v66p = R65(v55p + w55p x P56 + w55 x (w55 x P56)) + 2*(w66 x d6p*Z6_u) + d6pp*Z6_u: ')
v66p = R65 * (v55p + w55p.cross(P56) + w55.cross(w55.cross(P56))) + 2*(w66.cross(d6p[0] * z_u)) + d6pp[0] * z_u
v66p_simpl = R65 * (v55p_xyz + w55p_xyz.cross(P56) + w55_xyz.cross(w55_xyz.cross(P56))) + 2*(w66_xyz.cross(d6p[0] * z_u)) + d6pp[0] * z_u
print('v66p: ', v66p)
print('v66p_simpl: ', v66p_simpl)
v66p_xyz = sp.Matrix([vel66x_p, vel66y_p, vel66z_p])
print('vc66p = v66p + w66p x P6c6 + w66 x (w66 x P6c6)')
vc66p = v66p + w66p.cross(P6c6) + w66.cross(w66.cross(P6c6))
vc66p_simpl = v66p_xyz + w66p_xyz.cross(P6c6) + w66_xyz.cross(w66.cross(P6c6))
print('vc66p: ', vc66p)
# pprint(vc66p, use_unicode=False)
print('vc66p_simpl: ', vc66p_simpl)
print()

# Force
print('Force')
F1 = m1*vc11p
N1 = Ic1*w11p + w11.cross(Ic1*w11)
F2 = m2*vc22p
N2 = Ic2*w22p + w22.cross(Ic2*w22)
F3 = m3*vc33p
N3 = Ic3*w33p + w33.cross(Ic3*w33)
F4 = m4*vc44p
N4 = Ic4*w44p + w44.cross(Ic4*w44)
F5 = m5*vc55p
N5 = Ic5*w55p + w55.cross(Ic5*w55)
F6 = m6*vc66p
N6 = Ic6*w66p + w66.cross(Ic6*w66)
print('F1 = ', F1)
print('N1 = ', N1)
print('F2 = ', F2)
print('N2 = ', N2)
print('F3 = ', F3)
print('N3 = ', N3)
print('F4 = ', F4)
print('N4 = ', N4)
print('F5 = ', F5)
print('N5 = ', N5)
print('F6 = ', F6)
print('N6 = ', N6)
print()
# Outbound iterations
f6 = sp.Matrix([[0], [0], [0]])
n6 = sp.Matrix([[0], [0], [0]])
f5 = R56*f6+F5
n5 = N5 + R56*n6 + P5c5.cross(F5) + P56.cross(R56*f6)
f4 = R45*f5+F4
n4 = N4 + R45*n5 + P4c4.cross(F4) + P45.cross(R45*f5)
f3 = R34*f4+F3
n3 = N3 + R34*n4 + P3c3.cross(F3) + P34.cross(R34*f4)
f2 = R23*f3+F2
n2 = N2 + R23*n3 + P2c2.cross(F2) + P23.cross(R23*f3)
f1 = R12*f2+F1
n1 = N1 + R12*n2 + P1c1.cross(F1) + P12.cross(R12*f2)
print('f6 = ', f6)
print('n6 = ', n6)
print('f5 = ', f5)
print('n5 = ', n5)
print('f4 = ', f4)
print('n4 = ', n4)
print('f3 = ', f3)
print('n3 = ', n3)
print('f2 = ', f2)
print('n2 = ', n2)
print('f1 = ', f1)
print('n1 = ', n1)

i=0
time = []
n1_solz = []
n3_solz = []
f5_solz = []
for t in np.linspace(0, 3, 25):
    #
    # Motor 1
    t10 = 0
    t1f = 3
    Pos10 = 0
    Pos1f = invKin[0][0]
    Vel10 = 0
    Vel1f = 0
    acc10 = 0
    acc1f = 0
    # Coefficients
    coef_a10 = Pos10
    coef_a11 = Vel10
    coef_a12 = coef_a10/2
    coef_a13 = (20*Pos1f-20*Pos10-(8*Vel1f+12*Vel10)*t1f-(3*acc10-acc1f)*t1f**2)/(2*t1f**3)
    coef_a14 = (30*Pos10-30*Pos1f-(14*Vel1f+16*Vel10)*t1f+(3*acc10-2*acc1f)*t1f**2)/(2*t1f**4)
    coef_a15 = (12*Pos1f-12*Pos10-(6*Vel1f+6*Vel10)*t1f-(acc10-acc1f)*t1f**2)/(2*t1f**5)
    # Polynomial equation evaluation
    th1_res = coef_a10 + coef_a11*t + coef_a12*t**2 + coef_a13*t**3 + coef_a14*t**4 + coef_a15*t**5
    th1_d_res = coef_a11 + 2*coef_a12*t + 3*coef_a13*t**2 + 4*coef_a14*t**3 + 5*coef_a15*t**4
    th1_dd_res = 2*coef_a12 + 6*coef_a13*t + 12*coef_a14*t**2 + 20*coef_a15*t**3

    # Motor 2
    t20 = 0
    t2f = 3
    Pos20 = 0
    Pos2f = invKin[0][1]
    Vel20 = 0
    Vel2f = 0
    acc20 = 0
    acc2f = 0
    # Coefficients
    coef_a20 = Pos20
    coef_a21 = Vel20
    coef_a22 = coef_a20/2
    coef_a23 = (20*Pos2f-20*Pos20-(8*Vel2f+12*Vel20)*t2f-(3*acc20-acc2f)*t2f**2)/(2*t2f**3)
    coef_a24 = (30*Pos20-30*Pos2f-(14*Vel2f+16*Vel20)*t2f+(3*acc20-2*acc2f)*t2f**2)/(2*t2f**4)
    coef_a25 = (12*Pos2f-12*Pos20-(6*Vel2f+6*Vel20)*t2f-(acc20-acc2f)*t2f**2)/(2*t2f**5)
    # Polynomial equation evaluation
    th2_res = coef_a20 + coef_a21*t + coef_a22*t**2 + coef_a23*t**3 + coef_a24*t**4 + coef_a25*t**5
    th2_d_res = coef_a21 + 2*coef_a22*t + 3*coef_a23*t**2 + 4*coef_a24*t**3 + 5*coef_a25*t**4
    th2_dd_res = 2*coef_a22 + 6*coef_a23*t + 12*coef_a24*t**2 + 20*coef_a25*t**3

    # Motor 3
    t30 = 0
    t3f = 3
    Pos30 = 0
    Pos3f = invKin[0][2]
    Vel30 = 0
    Vel3f = 0
    acc30 = 0
    acc3f = 0
    # Coefficients
    coef_a30 = Pos30
    coef_a31 = Vel30
    coef_a32 = coef_a30/2
    coef_a33 = (20*Pos3f-20*Pos30-(8*Vel3f+12*Vel30)*t3f-(3*acc30-acc3f)*t3f**2)/(2*t3f**3)
    coef_a34 = (30*Pos30-30*Pos3f-(14*Vel3f+16*Vel30)*t3f+(3*acc30-2*acc3f)*t3f**2)/(2*t3f**4)
    coef_a35 = (12*Pos3f-12*Pos30-(6*Vel3f+6*Vel30)*t3f-(acc30-acc3f)*t3f**2)/(2*t3f**5)
    # Polynomial equation evaluation
    d3_res = coef_a30 + coef_a31*t + coef_a32*t**2 + coef_a33*t**3 + coef_a34*t**4 + coef_a35*t**5
    d3_d_res = coef_a31 + 2*coef_a32*t + 3*coef_a33*t**2 + 4*coef_a34*t**3 + 5*coef_a35*t**4
    d3_dd_res = 2*coef_a32 + 6*coef_a33*t + 12*coef_a34*t**2 + 20*coef_a35*t**3

    # Evaluation of outbound iterations
    # i=i+1
    n1_sol = n1.subs({th1:th1_res, th1_p: th1_d_res, th1_pp: th1_dd_res, th2:th2_res, th2_p: th2_d_res, th2_pp: th2_dd_res, d3:d3_res, d3_p: d3_d_res, d3_pp: d3_dd_res, L1: 0.75, L2: 0.40, a1: 0.15, L3: 0.40, a2: 0.40})
    n3_sol = n3.subs({th1:th1_res, th1_p: th1_d_res, th1_pp: th1_dd_res, th2:th2_res, th2_p: th2_d_res, th2_pp: th2_dd_res, d3:d3_res, d3_p: d3_d_res, d3_pp: d3_dd_res, L1: 0.75, L2: 0.40, a1: 0.15, L3: 0.40, a2: 0.40})
    f5_sol = f5.subs({th1:th1_res, th1_p: th1_d_res, th1_pp: th1_dd_res, th2:th2_res, th2_p: th2_d_res, th2_pp: th2_dd_res, d3:d3_res, d3_p: d3_d_res, d3_pp: d3_dd_res, L1: 0.75, L2: 0.40, a1: 0.15, L3: 0.40, a2: 0.40})
    n1_solz.append((n1_sol.T)*z_u)
    n3_solz.append((n3_sol.T)*z_u)
    f5_solz.append((f5_sol.T)*z_u)
    # print('\nn1_sol = ', n1_solz)
    # print('\nn3_sol = ', n3_solz)
    # print('\nn5_sol = ', n5_solz)
    time.append(t)

n1_lst = []
n3_lst = []
f5_lst = []
for i in range(len(n1_solz)):
    n1_lst.append(n1_solz[i][0])
    n3_lst.append(n3_solz[i][0])
    f5_lst.append(f5_solz[i][0])

# print(time)
# print(n1_solz)
# print(len(time))
# print(n1_solz[0][0])
# print(n1_solz[1])
# print(n1_solz[2])
# print(len(n1_solz))
fig2, axs = plt.subplots(3, 1, figsize=(9, 6), sharex=True)

axs[0].plot(time, n1_lst)
axs[0].set_xlabel('time [s]')
axs[0].set_ylabel('torque [kg m^2/s^2]')
axs[0].set_title('Motor Torque 1')
axs[0].grid(True)
axs[1].plot(time, n3_lst)
axs[1].set_xlabel('time [s]')
axs[1].set_ylabel('torque [kg m^2/s^2]')
axs[1].set_title('Motor Torque 2')
axs[1].grid(True)
axs[2].plot(time, f5_lst)
axs[2].set_xlabel('time [s]')
axs[2].set_ylabel('torque [kg m^2/s^2]')
axs[2].set_title('Motor Force 3')
axs[2].grid(True)
plt.tight_layout()
plt.show()

# Jacobian
print('\nJacobian')
Func1 = nsimplify(T06_ns[0, 3], tolerance=1e-10, rational=True)
Func2 = nsimplify(T06_ns[1, 3], tolerance=1e-10, rational=True)
Func3 = nsimplify(T06_ns[2, 3], tolerance=1e-10, rational=True)
print('Func1 = ',Func1)
print()
print('Func2 = ',Func2)
print()
print('Func3 = ',Func3)
print()
dF1dth1 = sp.diff(Func1, dof[0])
dF1dth2 = sp.diff(Func1, dof[1])
dF1dth3 = sp.diff(Func1, dof[2])
dF2dth1 = sp.diff(Func2, dof[0])
dF2dth2 = sp.diff(Func2, dof[1])
dF2dth3 = sp.diff(Func2, dof[2])
dF3dth1 = sp.diff(Func3, dof[0])
dF3dth2 = sp.diff(Func3, dof[1])
dF3dth3 = sp.diff(Func3, dof[2])
print('Partial derivates of Func1')
print(dF1dth1)
print(dF1dth2)
print(dF1dth3)
print('\nPartial derivates of Func2')
print(dF2dth1)
print(dF2dth2)
print(dF2dth3)
print('\nPartial derivates of Func3')
print(dF3dth1)
print(dF3dth2)
print(dF3dth3)
print()
J0 = sp.Matrix([
                [dF1dth1,dF1dth2,dF1dth3],
                [dF2dth1,dF2dth2,dF2dth3],
                [dF3dth1,dF3dth2,dF3dth3]
                ])
print('J0\n',J0)
# pprint(J0, use_unicode=False)

print('\nSecond way to calculate Jacobian from v66')
Func1 = nsimplify(v66[0], tolerance=1e-10, rational=True)
Func2 = nsimplify(v66[1], tolerance=1e-10, rational=True)
Func3 = nsimplify(v66[2], tolerance=1e-10, rational=True)
print('Func1 = ',sp.expand(Func1))
print()
print('Func2 = ',sp.expand(Func2))
print()
print('Func3 = ',sp.expand(Func3))
print()

# # 1
# J4 = sp.Matrix([
#                 [0,L3*sin(th3),0],
#                 [0,L3*cos(th3) + L4, L4],
#                 [-L3*cos(th2) - L4*cos(th2+th3),0,0]
#                 ])

# # 2
# J4 = sp.Matrix([
#                 [0,0,0],
#                 [-L3*cos(th3) + a1*sin(th3) - a4*sin(th3)**2 - a4*cos(th3)**2 - d2*cos(th3), - sin(th3), 0],
#                 [-L3*sin(th3) - a1*cos(th3) - d2*sin(th3),cos(th3),0]
#                 ])

# 3
# J4 = sp.Matrix([
#                 [L1*sin(th2+th3)+ L2*sin(th3),L2*sin(th3),0],
#                 [L1*cos(th2+th3) + L2*cos(th3) + L3,L2*cos(th3) + L3, L3],
#                 [0,0,0]
#                 ])

# # 4
# J4 = sp.Matrix([
#                 [0,0,1],
#                 [-a2*sin(th2) + a3*cos(th2) + d3*cos(th2), 0, 0],
#                 [0,0,0]
#                 ])

# # SCARA
J6 = sp.Matrix([
                [L2*sin(th2),0,0],
                [-L2*cos(th2) - L3, - L3, 0],
                [0,0,1]
                ])

print('J6\n',J6)
print()
R06 = nsimplify(T06_ns[:3, :3], tolerance=1e-10, rational=True)
print("To calculate Jacobian seen from 0 we need to multiply by corresponding rotational matrix")
print('R06: ', R06)
print('\nJ0 = R06*J6')
J0_ = R06*J6
print(J0_)
print("\nJ0 from the first method is equal to J0 from the second?: ",J0.equals(J0_))