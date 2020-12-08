import numpy as np
import sympy as sp
import math
from sympy import sin, cos, acos
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from tabulate import tabulate
from matplotlib.widgets import TextBox

Writer = animation.writers['ffmpeg']
writer = Writer(fps = 4, metadata=dict(artist='OscarSantos98'), bitrate=1800)

# Symbolic
a10, a11, a12, a13, a14, a15, a1, a2, a3, a4, a5, a6 = sp.symbols(
    'a10 a11 a12 a13 a14 a15 a1 a2 a3 a4 a5 a6')
th1, th2, th3, th4, th5, th6, d1, d2, d3, d4, d5, d6 = sp.symbols(
    'th1 th2 th3 th4 th5 th6 d1 d2 d3 d4 d5 d6')
L1, L2, L3, L4 = sp.symbols('L1 L2 L3 L4')

# Numeric data from SCARA
# Length of the links
_L1_ = 0.75 # in meters
_L2_ = 0.40 # in meters
_L3_ = 0.40 # in meters
_a1_ = 0.15 # in meters
# _d3_ = # in meters
_a2_ = 0.40 # in meters

# Inverse kinematics
x = float(input('Coordinate x (in cm): '))
y = float(input('Coordinate y (in cm): '))
z = float(input('Coordinate z (in cm): '))
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
print(f'{chr(945)} = arc cos[(q^2 + L2^2 - L3^2 )/(2*L2*q)] = ', alpha, '=', alpha_sold)
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
invKin = [[theta1r, theta2r, d3_sol],[theta1d, theta2d, d3_sol]]
print(tabulate(invKin, headers=[chr(952)+'1', chr(952)+'2', 'd3']))
print()
relationj = theta2d/theta1d
relationk = theta2d/(d3_sol*100)

a1_v = sp.Matrix([a10, a11, a12, a13, a14, a15])
a_v = sp.Matrix([a1, a2, a3, a4, a5, a6])
th_v = sp.Matrix([th1, th2, th3, th4, th5, th6])
d_v = sp.Matrix([d1, d2, d3, d4, d5, d6])

xmin = -50
xmax = 80
ymin = -60
ymax = 60
zmin = -2
zmax = 102

size = 7
# Define unit vectors
i0 = sp.Matrix([1, 0, 0, 1])
j0 = sp.Matrix([0, 1, 0, 1])
k0 = sp.Matrix([0, 0, 1, 1])

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.quiver(0, 0, 0, size * (i0[0]), size * (i0[1]),
          size * (i0[2]), color='red', linewidth=3)
ax.quiver(0, 0, 0, size * (j0[0]), size * (j0[1]),
          size * (j0[2]), color='green', linewidth=3)
ax.quiver(0, 0, 0, size * (k0[0]), size * (k0[1]),
          size * (k0[2]), color='blue', linewidth=3)

ax.set_title('3D Plot SCARA')
ax.set_xlabel('X_0 (cm)')
ax.set_ylabel('Y_0 (cm)')
ax.set_zlabel('Z_0 (cm)')
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_zlim(zmin, zmax)

# # line, = ax.plot(0, 0)

# # [1]
# # line.plot3D(np.linspace(float(0), float(T01[0, 3])), np.linspace(float(0), float(T01[1, 3])), np.linspace(float(0), float(T01[2, 3])))

def animation_frame(i):
    j = i/relationj
    k = i/relationk
    print(f'\n{chr(952)}1 = ',j)
    print(f'{chr(952)}2 = ',i)
    print('d3 = ',k)
    # print(i)
    # print(animation_frame.line_1)
    # print(animation_frame.line_11)
    # print(animation_frame.line_1[0])
    # print(animation_frame.line_2[0])
    # print(animation_frame.line_3[0])
    # print(animation_frame.line_4[0])
    # print(animation_frame.line_5[0])
    # print(animation_frame.line_6[0])
    animation_frame.line_1.pop(0).remove()
    animation_frame.line_11.remove()
    animation_frame.line_12.remove()
    animation_frame.line_13.remove()
    animation_frame.line_2.pop(0).remove()
    animation_frame.line_21.remove()
    animation_frame.line_22.remove()
    animation_frame.line_23.remove()
    animation_frame.line_3.pop(0).remove()
    animation_frame.line_31.remove()
    animation_frame.line_32.remove()
    animation_frame.line_33.remove()
    animation_frame.line_4.pop(0).remove()
    animation_frame.line_41.remove()
    animation_frame.line_42.remove()
    animation_frame.line_43.remove()
    animation_frame.line_5.pop(0).remove()
    animation_frame.line_51.remove()
    animation_frame.line_52.remove()
    animation_frame.line_53.remove()
    animation_frame.line_6.pop(0).remove()
    animation_frame.line_61.remove()
    animation_frame.line_62.remove()
    animation_frame.line_63.remove()
    # line1 = animation_frame.line_1.pop(0)
    # line2 = animation_frame.line_2.pop(0)
    # line3 = animation_frame.line_3.pop(0)
    # line4 = animation_frame.line_4.pop(0)
    # line5 = animation_frame.line_5.pop(0)
    # line6 = animation_frame.line_6.pop(0)
    # line1.remove()
    # line2.remove()
    # line3.remove()
    # line4.remove()
    # line5.remove()
    # line6.remove()

    data3 = [
        [1, 0, 0, 75, j * np.pi / 180],
        [2, 0, 40, 0, 0],
        [3, 0, 0, 15, i * np.pi / 180],
        [4, np.pi, 40, 0, 0],
        [5, 0, 0, k, 0],
        [6, 0, 0, 40, 0]
    ]
    # Numeric for the plot
    a1_n = a1_v.subs([(a10, data3[0][1]), (a11, data3[1][1]), (a12, data3[2][1]),
                      (a13, data3[3][1]), (a14, data3[4][1]), (a15, data3[5][1])])
    a_n = a_v.subs([(a1, data3[0][2]), (a2, data3[1][2]), (a3, data3[2][2]),
                    (a4, data3[3][2]), (a5, data3[4][2]), (a6, data3[5][2])])
    d_n = d_v.subs([(d1, data3[0][3]), (d2, data3[1][3]), (d3, data3[2][3]),
                    (d4, data3[3][3]), (d5, data3[4][3]), (d6, data3[5][3])])
    th_n = th_v.subs([(th1, data3[0][4]), (th2, data3[1][4]), (th3, data3[2][4]),
                      (th4, data3[3][4]), (th5, data3[4][4]), (th6, data3[5][4])])

    T = []
    for i in range(6):
        T.append(sp.Matrix([
            [cos(th_n[i]), -sin(th_n[i]), 0, a_n[i]],
            [sin(th_n[i]) * cos(a1_n[i]), cos(th_n[i]) *
             cos(a1_n[i]), -sin(a1_n[i]), -sin(a1_n[i]) * d_n[i]],
            [sin(th_n[i]) * sin(a1_n[i]), cos(th_n[i]) *
             sin(a1_n[i]), cos(a1_n[i]), cos(a1_n[i]) * d_n[i]],
            [0, 0, 0, 1],
        ])
        )
    T01 = T[0]
    T02 = T01 * T[1]
    T03 = T02 * T[2]
    T04 = T03 * T[3]
    T05 = T04 * T[4]
    T06 = T05 * T[5]

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
    # [1]
    animation_frame.line_1 = ax.plot3D(np.linspace(float(0), float(T01[0, 3])), np.linspace(float(0), float(T01[1, 3])), np.linspace(float(0), float(T01[2, 3])), color='tab:gray', linewidth=3)
    animation_frame.line_11 = ax.quiver(T01[0,3], T01[1,3], T01[2,3], size*(i01[0] - T01[0,3]), size*(i01[1] - T01[1,3]), size*(i01[2] - T01[2,3]), color='red', linewidth=3)
    animation_frame.line_12 = ax.quiver(T01[0,3], T01[1,3], T01[2,3], size*(j01[0] - T01[0,3]), size*(j01[1] - T01[1,3]), size*(j01[2] - T01[2,3]), color='green', linewidth=3)
    animation_frame.line_13 = ax.quiver(T01[0,3], T01[1,3], T01[2,3], size*(k01[0] - T01[0,3]), size*(k01[1] - T01[1,3]), size*(k01[2] - T01[2,3]), color='blue', linewidth=3)
    # [2]
    animation_frame.line_2 = ax.plot3D(np.linspace(float(T01[0,3]), float(T02[0,3])), np.linspace(float(T01[1,3]), float(T02[1,3])), np.linspace(float(T01[2,3]), float(T02[2,3])), color='tab:orange', linewidth=3)
    animation_frame.line_21 = ax.quiver(T02[0,3], T02[1,3], T02[2,3], size*(i02[0] - T02[0,3]), size*(i02[1] - T02[1,3]), size*(i02[2] - T02[2,3]), color='red', linewidth=3)
    animation_frame.line_22 = ax.quiver(T02[0,3], T02[1,3], T02[2,3], size*(j02[0] - T02[0,3]), size*(j02[1] - T02[1,3]), size*(j02[2] - T02[2,3]), color='green', linewidth=3)
    animation_frame.line_23 = ax.quiver(T02[0,3], T02[1,3], T02[2,3], size*(k02[0] - T02[0,3]), size*(k02[1] - T02[1,3]), size*(k02[2] - T02[2,3]), color='blue', linewidth=3)
    # [3]
    animation_frame.line_3 = ax.plot3D(np.linspace(float(T02[0,3]), float(T03[0,3])), np.linspace(float(T02[1,3]), float(T03[1,3])), np.linspace(float(T02[2,3]), float(T03[2,3])), color='tab:orange', linewidth=3)
    animation_frame.line_31 = ax.quiver(T03[0,3], T03[1,3], T03[2,3], size*(i03[0] - T03[0,3]), size*(i03[1] - T03[1,3]), size*(i03[2] - T03[2,3]), color='red', linewidth=3)
    animation_frame.line_32 = ax.quiver(T03[0,3], T03[1,3], T03[2,3], size*(j03[0] - T03[0,3]), size*(j03[1] - T03[1,3]), size*(j03[2] - T03[2,3]), color='green', linewidth=3)
    animation_frame.line_33 = ax.quiver(T03[0,3], T03[1,3], T03[2,3], size*(k03[0] - T03[0,3]), size*(k03[1] - T03[1,3]), size*(k03[2] - T03[2,3]), color='blue', linewidth=3)
    # [4]
    animation_frame.line_4 = ax.plot3D(np.linspace(float(T03[0,3]), float(T04[0,3])), np.linspace(float(T03[1,3]), float(T04[1,3])), np.linspace(float(T03[2,3]), float(T04[2,3])), color='yellow', linewidth=3)
    animation_frame.line_41 = ax.quiver(T04[0,3], T04[1,3], T04[2,3], size*(i04[0] - T04[0,3]), size*(i04[1] - T04[1,3]), size*(i04[2] - T04[2,3]), color='red', linewidth=3)
    animation_frame.line_42 = ax.quiver(T04[0,3], T04[1,3], T04[2,3], size*(j04[0] - T04[0,3]), size*(j04[1] - T04[1,3]), size*(j04[2] - T04[2,3]), color='green', linewidth=3)
    animation_frame.line_43 = ax.quiver(T04[0,3], T04[1,3], T04[2,3], size*(k04[0] - T04[0,3]), size*(k04[1] - T04[1,3]), size*(k04[2] - T04[2,3]), color='blue', linewidth=3)
    # [5]
    animation_frame.line_5 = ax.plot3D(np.linspace(float(T04[0,3]), float(T05[0,3])), np.linspace(float(T04[1,3]), float(T05[1,3])), np.linspace(float(T04[2,3]), float(T05[2,3])), color='tab:brown', linewidth=3)
    animation_frame.line_51 = ax.quiver(T05[0,3], T05[1,3], T05[2,3], size*(i05[0] - T05[0,3]), size*(i05[1] - T05[1,3]), size*(i05[2] - T05[2,3]), color='red', linewidth=3)
    animation_frame.line_52 = ax.quiver(T05[0,3], T05[1,3], T05[2,3], size*(j05[0] - T05[0,3]), size*(j05[1] - T05[1,3]), size*(j05[2] - T05[2,3]), color='green', linewidth=3)
    animation_frame.line_53 = ax.quiver(T05[0,3], T05[1,3], T05[2,3], size*(k05[0] - T05[0,3]), size*(k05[1] - T05[1,3]), size*(k05[2] - T05[2,3]), color='blue', linewidth=3)
    # [6]
    animation_frame.line_6 = ax.plot3D(np.linspace(float(T05[0,3]), float(T06[0,3])), np.linspace(float(T05[1,3]), float(T06[1,3])), np.linspace(float(T05[2,3]), float(T06[2,3])), color='tab:olive', linewidth=3)
    animation_frame.line_61 = ax.quiver(T06[0,3], T06[1,3], T06[2,3], size*(i06[0] - T06[0,3]), size*(i06[1] - T06[1,3]), size*(i06[2] - T06[2,3]), color='red', linewidth=3)
    animation_frame.line_62 = ax.quiver(T06[0,3], T06[1,3], T06[2,3], size*(j06[0] - T06[0,3]), size*(j06[1] - T06[1,3]), size*(j06[2] - T06[2,3]), color='green', linewidth=3)
    animation_frame.line_63 = ax.quiver(T06[0,3], T06[1,3], T06[2,3], size*(k06[0] - T06[0,3]), size*(k06[1] - T06[1,3]), size*(k06[2] - T06[2,3]), color='blue', linewidth=3)
    return animation_frame.line_1,

# [1]
animation_frame.line_1 = ax.plot3D(np.linspace(float(0), float(0)), np.linspace(float(0), float(0)), np.linspace(float(0), float(0)))
animation_frame.line_11 = ax.quiver(0, 0, 0, 0, 0, 0)
animation_frame.line_12 = ax.quiver(0, 0, 0, 0, 0, 0)
animation_frame.line_13 = ax.quiver(0, 0, 0, 0, 0, 0)
# [2]
animation_frame.line_2 = ax.plot3D(np.linspace(float(0), float(0)), np.linspace(float(0), float(0)), np.linspace(float(0), float(0)))
animation_frame.line_21 = ax.quiver(0, 0, 0, 0, 0, 0)
animation_frame.line_22 = ax.quiver(0, 0, 0, 0, 0, 0)
animation_frame.line_23 = ax.quiver(0, 0, 0, 0, 0, 0)
# [3]
animation_frame.line_3 = ax.plot3D(np.linspace(float(0), float(0)), np.linspace(float(0), float(0)), np.linspace(float(0), float(0)))
animation_frame.line_31 = ax.quiver(0, 0, 0, 0, 0, 0)
animation_frame.line_32 = ax.quiver(0, 0, 0, 0, 0, 0)
animation_frame.line_33 = ax.quiver(0, 0, 0, 0, 0, 0)
# [4]
animation_frame.line_4 = ax.plot3D(np.linspace(float(0), float(0)), np.linspace(float(0), float(0)), np.linspace(float(0), float(0)))
animation_frame.line_41 = ax.quiver(0, 0, 0, 0, 0, 0)
animation_frame.line_42 = ax.quiver(0, 0, 0, 0, 0, 0)
animation_frame.line_43 = ax.quiver(0, 0, 0, 0, 0, 0)
# [5]
animation_frame.line_5 = ax.plot3D(np.linspace(float(0), float(0)), np.linspace(float(0), float(0)), np.linspace(float(0), float(0)))
animation_frame.line_51 = ax.quiver(0, 0, 0, 0, 0, 0)
animation_frame.line_52 = ax.quiver(0, 0, 0, 0, 0, 0)
animation_frame.line_53 = ax.quiver(0, 0, 0, 0, 0, 0)
# [6]
animation_frame.line_6 = ax.plot3D(np.linspace(float(0), float(0)), np.linspace(float(0), float(0)), np.linspace(float(0), float(0)))
animation_frame.line_61 = ax.quiver(0, 0, 0, 0, 0, 0)
animation_frame.line_62 = ax.quiver(0, 0, 0, 0, 0, 0)
animation_frame.line_63 = ax.quiver(0, 0, 0, 0, 0, 0)

animation = animation.FuncAnimation(fig, func=animation_frame,
                          frames=np.arange(0, theta2d+theta2d/100, theta2d/10), interval=250, blit=False, repeat=False)

textstr = '\n'.join((
    r'$\theta_1=%.2f°$' % (theta1d, ),
    r'$\theta_2=%.2f°$' % (theta2d, ),
    r'$d_3=%.2f$' % (d3_sol*100, )))
# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='lavender', alpha=0.5)

# place a text box in upper left in axes coords
ax.text(0, 0, 750, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

plt.subplots_adjust(bottom=0.2)
initial_text = f"Coordinates (x: {round(x*100,2)} ,y: {round(y*100,2)}, z: {round(z*100,2)})"
def submit(text):
    ydata = eval(text)
    ax.set_ydata(ydata)
    ax.set_ylim(np.min(ydata), np.max(ydata))
    plt.draw()

# axbox = plt.axes([0.1, 0.05, 0.8, 0.075])
axbox = plt.axes([0.3, 0.05, 0.45, 0.075])
text_box = TextBox(axbox, 'Evaluation', initial=initial_text)
text_box.on_submit(submit)

animation.save('Movement.mp4', writer=writer)
# plt.show()
