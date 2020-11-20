import numpy as np
import sympy as sp
from sympy import sin, cos
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

Writer = animation.writers['ffmpeg']
writer = Writer(fps = 3, metadata=dict(artist='OscarSantos98'), bitrate=1800)

# Symbolic
a10, a11, a12, a13, a14, a15, a1, a2, a3, a4, a5, a6 = sp.symbols(
    'a10 a11 a12 a13 a14 a15 a1 a2 a3 a4 a5 a6')
th1, th2, th3, th4, th5, th6, d1, d2, d3, d4, d5, d6 = sp.symbols(
    'th1 th2 th3 th4 th5 th6 d1 d2 d3 d4 d5 d6')

a1_v = sp.Matrix([a10, a11, a12, a13, a14, a15])
a_v = sp.Matrix([a1, a2, a3, a4, a5, a6])
th_v = sp.Matrix([th1, th2, th3, th4, th5, th6])
d_v = sp.Matrix([d1, d2, d3, d4, d5, d6])

data3 = [
    [1, 0, 0, 75, 95 * np.pi / 180],
    [2, 0, 40, 0, -70 * np.pi / 180],
    [3, 0, 0, 15, 0],
    [4, np.pi, 40, 0, 0],
    [5, 0, 0, 40, 0],
    [6, 0, 0, 40, 0]
]
xmin = -50
xmax = 50
ymin = -10
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

ax.set_title('3D Plot')
ax.set_xlabel('Eje X_0')
ax.set_ylabel('Eje Y_0')
ax.set_zlabel('Eje Z_0')
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_zlim(zmin, zmax)

# line, = ax.plot(0, 0)

# [1]
# line.plot3D(np.linspace(float(0), float(T01[0, 3])), np.linspace(float(0), float(T01[1, 3])), np.linspace(float(0), float(T01[2, 3])))
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


def animation_frame(i):
    j = 2.2*i
    # print(i)
    # print(animation_frame.line_1[0])
    # print(animation_frame.line_2[0])
    # print(animation_frame.line_3[0])
    # print(animation_frame.line_4[0])
    # print(animation_frame.line_5[0])
    # print(animation_frame.line_6[0])
    animation_frame.line_1.pop(0).remove()
    animation_frame.line_2.pop(0).remove()
    animation_frame.line_3.pop(0).remove()
    animation_frame.line_4.pop(0).remove()
    animation_frame.line_5.pop(0).remove()
    animation_frame.line_6.pop(0).remove()
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
        [1, 0, 0, 75, (-i/2 + 65) * np.pi / 180],
        [2, 0, 40, 0, -(-2*j+70) * np.pi / 180],
        [3, 0, 0, 15, 0],
        [4, np.pi, 40, 0, 0],
        [5, 0, 0, i, 0],
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
    # ax.quiver(T01[0,3], T01[1,3], T01[2,3], size*(i01[0] - T01[0,3]), size*(i01[1] - T01[1,3]), size*(i01[2] - T01[2,3]), color='red', linewidth=3)
    # ax.quiver(T01[0,3], T01[1,3], T01[2,3], size*(j01[0] - T01[0,3]), size*(j01[1] - T01[1,3]), size*(j01[2] - T01[2,3]), color='green', linewidth=3)
    # ax.quiver(T01[0,3], T01[1,3], T01[2,3], size*(k01[0] - T01[0,3]), size*(k01[1] - T01[1,3]), size*(k01[2] - T01[2,3]), color='blue', linewidth=3)

    # [2]
    animation_frame.line_2 = ax.plot3D(np.linspace(float(T01[0,3]), float(T02[0,3])), np.linspace(float(T01[1,3]), float(T02[1,3])), np.linspace(float(T01[2,3]), float(T02[2,3])), color='tab:orange', linewidth=3)
    # ax.quiver(T02[0,3], T02[1,3], T02[2,3], size*(i02[0] - T02[0,3]), size*(i02[1] - T02[1,3]), size*(i02[2] - T02[2,3]), color='red', linewidth=3)
    # ax.quiver(T02[0,3], T02[1,3], T02[2,3], size*(j02[0] - T02[0,3]), size*(j02[1] - T02[1,3]), size*(j02[2] - T02[2,3]), color='green', linewidth=3)
    # ax.quiver(T02[0,3], T02[1,3], T02[2,3], size*(k02[0] - T02[0,3]), size*(k02[1] - T02[1,3]), size*(k02[2] - T02[2,3]), color='blue', linewidth=3)
    # [3]
    animation_frame.line_3 = ax.plot3D(np.linspace(float(T02[0,3]), float(T03[0,3])), np.linspace(float(T02[1,3]), float(T03[1,3])), np.linspace(float(T02[2,3]), float(T03[2,3])), color='tab:orange', linewidth=3)
    # ax.quiver(T03[0,3], T03[1,3], T03[2,3], size*(i03[0] - T03[0,3]), size*(i03[1] - T03[1,3]), size*(i03[2] - T03[2,3]), color='red', linewidth=3)
    # ax.quiver(T03[0,3], T03[1,3], T03[2,3], size*(j03[0] - T03[0,3]), size*(j03[1] - T03[1,3]), size*(j03[2] - T03[2,3]), color='green', linewidth=3)
    # ax.quiver(T03[0,3], T03[1,3], T03[2,3], size*(k03[0] - T03[0,3]), size*(k03[1] - T03[1,3]), size*(k03[2] - T03[2,3]), color='blue', linewidth=3)
    # [4]
    animation_frame.line_4 = ax.plot3D(np.linspace(float(T03[0,3]), float(T04[0,3])), np.linspace(float(T03[1,3]), float(T04[1,3])), np.linspace(float(T03[2,3]), float(T04[2,3])), color='yellow', linewidth=3)
    # ax.quiver(T04[0,3], T04[1,3], T04[2,3], size*(i04[0] - T04[0,3]), size*(i04[1] - T04[1,3]), size*(i04[2] - T04[2,3]), color='red', linewidth=3)
    # ax.quiver(T04[0,3], T04[1,3], T04[2,3], size*(j04[0] - T04[0,3]), size*(j04[1] - T04[1,3]), size*(j04[2] - T04[2,3]), color='green', linewidth=3)
    # ax.quiver(T04[0,3], T04[1,3], T04[2,3], size*(k04[0] - T04[0,3]), size*(k04[1] - T04[1,3]), size*(k04[2] - T04[2,3]), color='blue', linewidth=3)
    # [5]
    animation_frame.line_5 = ax.plot3D(np.linspace(float(T04[0,3]), float(T05[0,3])), np.linspace(float(T04[1,3]), float(T05[1,3])), np.linspace(float(T04[2,3]), float(T05[2,3])), color='tab:gray', linewidth=3)
    # ax.quiver(T05[0,3], T05[1,3], T05[2,3], size*(i05[0] - T05[0,3]), size*(i05[1] - T05[1,3]), size*(i05[2] - T05[2,3]), color='red', linewidth=3)
    # ax.quiver(T05[0,3], T05[1,3], T05[2,3], size*(j05[0] - T05[0,3]), size*(j05[1] - T05[1,3]), size*(j05[2] - T05[2,3]), color='green', linewidth=3)
    # ax.quiver(T05[0,3], T05[1,3], T05[2,3], size*(k05[0] - T05[0,3]), size*(k05[1] - T05[1,3]), size*(k05[2] - T05[2,3]), color='blue', linewidth=3)
    # [6]
    animation_frame.line_6 = ax.plot3D(np.linspace(float(T05[0,3]), float(T06[0,3])), np.linspace(float(T05[1,3]), float(T06[1,3])), np.linspace(float(T05[2,3]), float(T06[2,3])), color='k', linewidth=3)
    # ax.quiver(T06[0,3], T06[1,3], T06[2,3], size*(i06[0] - T06[0,3]), size*(i06[1] - T06[1,3]), size*(i06[2] - T06[2,3]), color='red', linewidth=3)
    # ax.quiver(T06[0,3], T06[1,3], T06[2,3], size*(j06[0] - T06[0,3]), size*(j06[1] - T06[1,3]), size*(j06[2] - T06[2,3]), color='green', linewidth=3)
    # ax.quiver(T06[0,3], T06[1,3], T06[2,3], size*(k06[0] - T06[0,3]), size*(k06[1] - T06[1,3]), size*(k06[2] - T06[2,3]), color='blue', linewidth=3)
    return animation_frame.line_2,


animation_frame.line_1 = ax.plot3D(np.linspace(float(0), float(T01[0, 3])), np.linspace(float(0), float(T01[1, 3])), np.linspace(float(0), float(T01[2, 3])), color='tab:gray', linewidth=3)
animation_frame.line_2 = ax.plot3D(np.linspace(float(T01[0,3]), float(T02[0,3])), np.linspace(float(T01[1,3]), float(T02[1,3])), np.linspace(float(T01[2,3]), float(T02[2,3])), color='tab:orange', linewidth=3)
animation_frame.line_3 = ax.plot3D(np.linspace(float(T02[0,3]), float(T03[0,3])), np.linspace(float(T02[1,3]), float(T03[1,3])), np.linspace(float(T02[2,3]), float(T03[2,3])), color='tab:orange', linewidth=3)
animation_frame.line_4 = ax.plot3D(np.linspace(float(T03[0,3]), float(T04[0,3])), np.linspace(float(T03[1,3]), float(T04[1,3])), np.linspace(float(T03[2,3]), float(T04[2,3])), color='yellow', linewidth=3)
animation_frame.line_5 = ax.plot3D(np.linspace(float(T04[0,3]), float(T05[0,3])), np.linspace(float(T04[1,3]), float(T05[1,3])), np.linspace(float(T04[2,3]), float(T05[2,3])), color='tab:gray', linewidth=3)
animation_frame.line_6 = ax.plot3D(np.linspace(float(T05[0,3]), float(T06[0,3])), np.linspace(float(T05[1,3]), float(T06[1,3])), np.linspace(float(T05[2,3]), float(T06[2,3])), color='k', linewidth=3)

animation = animation.FuncAnimation(fig, func=animation_frame,
                          frames=np.arange(0, 50, 5), interval=250, blit=False, repeat=False)

# ax.quiver(T01[0,3], T01[1,3], T01[2,3], size*(i01[0] - T01[0,3]), size*(i01[1] - T01[1,3]), size*(i01[2] - T01[2,3]), color='red', linewidth=3)
# ax.quiver(T01[0,3], T01[1,3], T01[2,3], size*(j01[0] - T01[0,3]), size*(j01[1] - T01[1,3]), size*(j01[2] - T01[2,3]), color='green', linewidth=3)
# ax.quiver(T01[0,3], T01[1,3], T01[2,3], size*(k01[0] - T01[0,3]), size*(k01[1] - T01[1,3]), size*(k01[2] - T01[2,3]), color='blue', linewidth=3)
# ax.quiver(T02[0,3], T02[1,3], T02[2,3], size*(i02[0] - T02[0,3]), size*(i02[1] - T02[1,3]), size*(i02[2] - T02[2,3]), color='red', linewidth=3)
# ax.quiver(T02[0,3], T02[1,3], T02[2,3], size*(j02[0] - T02[0,3]), size*(j02[1] - T02[1,3]), size*(j02[2] - T02[2,3]), color='green', linewidth=3)
# ax.quiver(T02[0,3], T02[1,3], T02[2,3], size*(k02[0] - T02[0,3]), size*(k02[1] - T02[1,3]), size*(k02[2] - T02[2,3]), color='blue', linewidth=3)
# ax.quiver(T03[0,3], T03[1,3], T03[2,3], size*(i03[0] - T03[0,3]), size*(i03[1] - T03[1,3]), size*(i03[2] - T03[2,3]), color='red', linewidth=3)
# ax.quiver(T03[0,3], T03[1,3], T03[2,3], size*(j03[0] - T03[0,3]), size*(j03[1] - T03[1,3]), size*(j03[2] - T03[2,3]), color='green', linewidth=3)
# ax.quiver(T03[0,3], T03[1,3], T03[2,3], size*(k03[0] - T03[0,3]), size*(k03[1] - T03[1,3]), size*(k03[2] - T03[2,3]), color='blue', linewidth=3)
# ax.quiver(T04[0,3], T04[1,3], T04[2,3], size*(i04[0] - T04[0,3]), size*(i04[1] - T04[1,3]), size*(i04[2] - T04[2,3]), color='red', linewidth=3)
# ax.quiver(T04[0,3], T04[1,3], T04[2,3], size*(j04[0] - T04[0,3]), size*(j04[1] - T04[1,3]), size*(j04[2] - T04[2,3]), color='green', linewidth=3)
# ax.quiver(T04[0,3], T04[1,3], T04[2,3], size*(k04[0] - T04[0,3]), size*(k04[1] - T04[1,3]), size*(k04[2] - T04[2,3]), color='blue', linewidth=3)
# ax.quiver(T05[0,3], T05[1,3], T05[2,3], size*(i05[0] - T05[0,3]), size*(i05[1] - T05[1,3]), size*(i05[2] - T05[2,3]), color='red', linewidth=3)
# ax.quiver(T05[0,3], T05[1,3], T05[2,3], size*(j05[0] - T05[0,3]), size*(j05[1] - T05[1,3]), size*(j05[2] - T05[2,3]), color='green', linewidth=3)
# ax.quiver(T05[0,3], T05[1,3], T05[2,3], size*(k05[0] - T05[0,3]), size*(k05[1] - T05[1,3]), size*(k05[2] - T05[2,3]), color='blue', linewidth=3)
# ax.quiver(T06[0,3], T06[1,3], T06[2,3], size*(i06[0] - T06[0,3]), size*(i06[1] - T06[1,3]), size*(i06[2] - T06[2,3]), color='red', linewidth=3)
# ax.quiver(T06[0,3], T06[1,3], T06[2,3], size*(j06[0] - T06[0,3]), size*(j06[1] - T06[1,3]), size*(j06[2] - T06[2,3]), color='green', linewidth=3)
# ax.quiver(T06[0,3], T06[1,3], T06[2,3], size*(k06[0] - T06[0,3]), size*(k06[1] - T06[1,3]), size*(k06[2] - T06[2,3]), color='blue', linewidth=3)

# animation.save('Movimiento2.mp4', writer=writer)
plt.show()
