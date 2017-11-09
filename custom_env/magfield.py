"""
Inputs:
    Fx: Force on agent in x axis
    Fy: Force on agent in y axis
    x: agent position from center in x
    y: agent position from center in y
Function to plot 2D vector plots representing the magnetic
field created by two orthoganally arranged paired coils.

Author: JF Lafleche

Reference equations:
http://web.mit.edu/viz/EM/visualizations/coursenotes/modules/guide09.pdf

"""

import numpy as np
import math
import matplotlib.pyplot as plt
from bokeh.layouts import column
from bokeh.plotting import figure, show, output_file
import time

class MagField():
    def __init__(self, R=0.1, l=0.1, N=44):
        self.R = R # m
        self.l = l # m
        self.N = N
        self.U_0 = 4*math.pi*10**(-7)


    def plot(self, Fx, Fy, x, y, plotter='mpl'):
        ###############################################
        # TO BE REPLACED BY FINAL IMPLEMENTATION
        if Fx == 0 and Fy == 0:
            I1x = 0
            I1y = 0
        elif Fx == 0:
            I1x = 0
            I1y = 5
        elif Fy == 0:
            I1x = 5
            I1y = 0
        else:
            I1x = 5
            I1y = I1x*(Fy/Fx)

        x_ratio = ((x-self.l/2)**2 + self.R**2)/((x+self.l/2)**2 + self.R**2)
        y_ratio = ((y-self.l/2)**2 + self.R**2)/((y+self.l/2)**2 + self.R**2)
        I2x = I1x*x_ratio
        I2y = I1y*y_ratio

        ###############################################

        # define square workspace size
        ws = np.linspace(-0.05, 0.05, 50)
        self.xx, self.yy = np.meshgrid(ws, ws)

        # u, v = _genMagField(R,5,-5,0,0,N,l)
        # u1, v1 = self._genGradField(I1x,I2x,I1y,I2y)
        u, v = self._gradientField_arr(I1x, I2x, I1y, I2y, self.xx, self.yy)

        # PLOT
        if plotter == 'mpl':
            self._matplt_quiver(u,v)
        elif plotter == 'bokeh':
            x0, y0, x1, y1, colors = self._bokeh_quiver(u,v)
            return x0, y0, x1, y1, colors

    def _magneticField(self, I_1, I_2, z):
        # eq. 9.9.1
        B = (0.5*self.U_0*self.N*self.R**2)*((I_1/((z-self.l/2)**2+self.R**2)**(3/2))+(I_2/((z+self.l/2)**2+self.R**2)**(3/2)))
        return B

    def _magneticField_arr(self, I1x, I2x, I1y, I2y, xx, yy):
        # eq 9.9.9
        array1_xx = xx - self.l/2
        array2_xx = xx + self.l/2
        array1_yy = yy - self.l/2
        array2_yy = yy + self.l/2

        term1_xx = np.divide(1,(array1_xx**2 + self.R**2)**(3/2))
        term2_xx = np.divide(1,(array2_xx**2 + self.R**2)**(3/2))
        term1_yy = np.divide(1,(array1_yy**2 + self.R**2)**(3/2))
        term2_yy = np.divide(1,(array2_yy**2 + self.R**2)**(3/2))
        Bpx = (0.5*self.U_0*self.N*self.R**2)*(I1x*term1_xx - I2x*term2_xx)
        Bpy = (0.5*self.U_0*self.N*self.R**2)*(I1y*term1_yy - I2y*term2_yy)
        return Bpx, Bpy

    def _gradientField_arr(self, I1x, I2x, I1y, I2y, xx, yy):
        # eq 9.9.9
        array1_xx = xx - self.l/2
        array2_xx = xx + self.l/2
        array1_yy = yy - self.l/2
        array2_yy = yy + self.l/2

        term1_xx = np.divide(3*array1_xx,(array1_xx**2 + self.R**2)**(5/2))
        term2_xx = np.divide(3*array2_xx,(array2_xx**2 + self.R**2)**(5/2))
        term1_yy = np.divide(3*array1_yy,(array1_yy**2 + self.R**2)**(5/2))
        term2_yy = np.divide(3*array2_yy,(array2_yy**2 + self.R**2)**(5/2))
        Bpx = (0.5*self.U_0*self.N*self.R**2)*(-I1x*term1_xx + I2x*term2_xx)
        Bpy = (0.5*self.U_0*self.N*self.R**2)*(-I1y*term1_yy + I2y*term2_yy)
        return Bpx, Bpy

    def _gradientField(self, I_1, I_2, z):
        # eq 9.9.9
        term1 = (I_1*3*(z-self.l/2))/((z-self.l/2)**2+self.R**2)**(5/2)
        term2 = (I_2*3*(z+self.l/2))/((z+self.l/2)**2+self.R**2)**(5/2)
        Bp = (0.5*self.U_0*self.N*self.R**2)*(-term1+term2)
        return Bp

    def _genMagField(self, I1_x, I2_x, I1_y, I2_y):
        magfield_u = np.zeros(self.xx.shape)
        magfield_v = np.zeros(self.yy.shape)

        for row_ix, row in enumerate(self.xx):
            for col_ix, pt in enumerate(row):
                B = self._magneticField(I1_x, I2_x, pt)
                magfield_u[row_ix, col_ix] = B

        for row_ix, row in enumerate(self.yy) :
            for col_ix, pt in enumerate(row):
                B = self._magneticField(I1_y, I2_y, pt)
                magfield_v[row_ix, col_ix] = B

        return magfield_u, magfield_v

    def _genGradField(self,I1_x, I2_x, I1_y, I2_y):
        # define square workspace size

        gradfield_u = np.zeros(self.xx.shape)
        gradfield_v = np.zeros(self.yy.shape)

        for row_ix, row in enumerate(self.xx):
            for col_ix, pt in enumerate(row):
                Bp = self._gradientField(I1_x, I2_x, pt)
                gradfield_u[row_ix, col_ix] = Bp

        for row_ix, row in enumerate(self.yy) :
            for col_ix, pt in enumerate(row):
                Bp = self._gradientField(I1_y, I2_y, pt)
                gradfield_v[row_ix, col_ix] = Bp

        return gradfield_u, gradfield_v

    def _matplt_quiver(self,u,v):
        plt.quiver(u,v)
        plt.show()

    def _bokeh_quiver(self,u,v):
        U = u+0.0001
        V = v+0.0002
        speed = np.sqrt(U*U + V*V)

        theta = np.arctan(V/U)

        x0 = self.xx[::2, ::2].flatten()
        y0 = self.yy[::2, ::2].flatten()
        length = speed[::2, ::2].flatten()/5
        angle = theta[::2, ::2].flatten()
        x1 = x0 + length * np.cos(angle)
        y1 = y0 + length * np.sin(angle)

        cm = np.array(["#E91E63", "#9C27B0", "#673AB7", "#3F51B5", "#03A9F4", "#00BCD4"])
        if length.min() == length.max():
            ix = np.ones((length.shape)).astype('int')
        else:
            ix = ((length-length.min())/(length.max()-length.min())*5).astype('int')
        if np.min(ix) == np.max(ix): ix = ix*0+5
        colors = cm[ix]

        return x0, y0, x1, y1, colors

        # self.p = figure()

        # self.p.segment(x0, y0, x1, y1, color=colors, line_width=2)

        # self.p.xaxis.visible = False
        # self.p.xgrid.visible = False
        # self.p.yaxis.visible = False
        # self.p.ygrid.visible = False

        # show(p)

magfield = MagField()

magfield.plot(0,0,0.05,0.05,plotter='bokeh')