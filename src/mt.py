'''
 *  Author : Bao Jiarong
 *  Contact: bao.salirong@gmail.com
 *  Created  On: 2020-05-28
 *  Modified On: 2020-05-28
 '''
from .algo import *

class MT(Algo):
    #--------------------------------------------------------
    # Constructor
    #---------------------------------------------------------
    def __init__(self,f,n,x0,y0,a,T,verbose=False):
        Algo.__init__(self,f,n,T,verbose)
        self.x0     =  x0
        self.y0     =  y0
        self.a      =  a

    #------------------------------------------------------
    def random_search(self,low,high):
        x = self.x0
        y = self.y0
        fmin = self.f(x,y)
        xmin = x
        ymin = y
        for i in range(self.n):
            x = randfloat(low,high)
            y = randfloat(low,high)
            if self.f(x,y) < fmin:
                fmin = self.f(x,y)
                xmin = x
                ymin = y
            if self.verbose == True and i % 10 == 0:
                print (i, round(fmin,5))
        return xmin,ymin

    #------------------------------------------------------
    def simple_descent(self):
        x = self.x0
        y = self.y0
        for i in range(self.n):
            x1 = randfloat(x-self.a,x+self.a)
            y1 = randfloat(y-self.a,y+self.a)
            if self.f(x1,y1)<self.f(x,y):
                x = x1
                y = y1
            if self.verbose == True and i % 10 == 0:
                print (i, round(self.f(x,y),5))
        return x,y

    #------------------------------------------------------
    def deepest_descent(self):
        x = self.x0
        y = self.y0
        for j in range(self.n):
            s1 = []
            s2 = []
            t1 = np.array([])
            t2 = np.array([])
            for i in range(self.a):
                s1.append(randfloat(x-self.a,x+self.a))
                s2.append(randfloat(y-self.a,y+self.a))
                t1 = np.append(t1,self.f(s1[i],s2[i]))

            index1 = t1.argmin()
            x1 = s1[index1]
            y1 = s2[index1]

            if self.f(x1,y1)<self.f(x,y):
                x = x1
                y = y1
            if self.verbose == True and j % 10 == 0:
                print (j, round(self.f(x,y),5))
        return x,y

    #------------------------------------------------------
    def multistart_descent(self):
        fbest = float('inf')
        best = float('inf')
        for i in range(self.n):
            self.x0 = randfloat(-self.a,self.a)
            self.y0 = randfloat(-self.a,self.a)
            x,y = self.deepest_descent()
            if self.f(x,y) < fbest:
                fbest = self.f(x,y)
                best = (x,y)
            if self.verbose == True and i % 10 == 0:
                print (i, round(self.f(best[0],best[1]),5))
        return x,y

    #------------------------------------------------------
    def tabu_search(self,L):
        fmin = self.f(self.x0,self.y0)
        xmin = self.x0
        ymin = self.y0
        tabu = np.array([])
        for j in range(self.n):
            s1 = np.array([])
            s2 = np.array([])
            t = np.array([])
            for i in range(self.a):
                r = randfloat(xmin-self.a,xmin+self.a)
                if r not in tabu:
                    s1 = np.append(s1,r)
                    s2 = np.append(s2,r)
                    t = np.append(t,(self.f(s1[i],s2[i])))
            index = t.argmin()
            x1 = s1[index]
            y1 = s2[index]
            tabu = np.append(tabu,x1)
            if tabu.shape[0] >= L:
                tabu = np.delete(tabu,0)
            if self.f(x1,y1)<fmin:
                fmin = self.f(x1,y1)
                xmin = x1
                ymin = y1
            if self.verbose == True and j % 10 == 0:
                print (j, round(self.f(xmin,ymin),5))
        return xmin,ymin

    #------------------------------------------------------
    def decrease_temperature(self,method="linear",alpha = 0.1,gamma = 3,delta = 4):
        if method == "linear":
            T = alpha * self.T
        if method == "discrete":
            T = self.T - alpha
        if method == "exponential":
            T = self.T * np.exp((-delta*self.T)/gamma)
        return T

    #------------------------------------------------------
    def metropolis_rule(self,x1,y1,eps=1e-3):
        delta = self.f(x1,y1) - self.f(self.x0,self.y0)
        if delta <= 0:
            x0 = x1
            y0 = y1
            return x0,y0
        else:
            p = np.exp(-delta/(self.T + eps))
            r = randfloat(0,1)
            if r <= p:
                x0 = x1
                y0 = y1
                return x0,y0
        return x1,y1

    #------------------------------------------------------
    def simulated_annealing(self,method="linear"):
        threshold = 4
        fmin = self.f(self.x0,self.y0)
        xmin = self.x0
        ymin = self.y0
        while self.T > threshold:
            for j in range(self.n):
                x1 = randfloat(xmin-10,xmin+10)
                y1 = randfloat(ymin-10,ymin+10)

                x0,y0 = self.metropolis_rule(x1,y1)

                if self.f(x0,y0) < fmin:
                    fmin = self.f(x0,y0)
                    xmin = x0
                    ymin = y0
                self.T = self.decrease_temperature()
                if self.verbose == True and j % 10 == 0:
                    print (j, round(self.f(xmin,ymin),5))
        return xmin,ymin

    #------------------------------------------------------
    def threshold_accept(self,method="linear"):
        moved = True
        while moved == True:
            for i in range(1,self.n):
                x = randfloat(self.x0-self.a,self.x0+self.a)
                y = randfloat(self.y0-self.a,self.y0+self.a)
                delta = self.f(x,y)-self.f(self.x0,self.y0)
                moved = True
                if delta < 0 or delta < self.T:
                    self.x0 = x
                    self.y0 = y
                else:
                    moved = False
                best = (self.x0,self.y0)

                if self.verbose == True and i % 10 == 0:
                    print (i, round(self.f(best[0],best[1]),5))
                self.T = self.decrease_temperature()
        return best
