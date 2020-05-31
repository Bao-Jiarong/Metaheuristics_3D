
'''
 *  Author : Bao Jiarong
 *  Contact: bao.salirong@gmail.com
 *  Created  On: 2020-05-28
 *  Modified On: 2020-05-28
 '''
import src.mt as meta
import numpy as np
import sys

def ackley(x,y):
    a = -20*np.exp(-0.2*math.sqrt(0.5*(x**2+y**2)))\
        -np.exp(0.5*(np.cos(2*np.pi*x)+(np.cos(2*np.pi*y))))+np.e+20
    return a

def beale(x,y):
    a = ((1.5-x+x*y)**2)+(2.25-x+x*(y**2))**2+(2.625-x+x*(y**3))**2
    return a

def goldstein_price(x,y):
    a = (1+((x+y+1)**2)*(19-14*x+3*(x**2)-14*y+6*x*y+3*(y**2)))\
        *(30+((2*x-3*y)**2)*(18-32*x+12*(x**2)+48*y-36*x*y+27*(y**2)))
    return a

def booth(x,y):
    a = ((x+2*y-7)**2)+(2*x+y-5)**2
    return a

def bukin(x,y):
    a = 100*math.sqrt(abs(y-0.01*(x**2)))+0.01*abs(x+10)
    return a

def matyas(x,y):
    a = 0.26*(x**2+y**2)-0.48*x*y
    return a

def levi(x,y):
    a = np.sin(3*np.pi*x)**2+((x-1)**2)*(1+np.sin(3*np.pi*y)**2)\
        +((y-1)**2)*(1+np.sin(2*np.pi*y)**2)
    return a

def himmelblau(x,y):
    a = (x**2+y-11)**2+(x+y**2-7)**2
    return a

def three_hump_camel(x,y):
    a = 2*x**2-1.05*x**4+((x**6)/6)+x*y+y**2
    return a

def easom(x,y):
    a = -np.cos(x)*np.cos(y)*np.exp(-((x-np.pi)**2+(y-np.pi)**2))
    return a

def cross_in_tray(x,y):
    a = -0.0001*(abs((np.sin(x)*np.sin(y)*np.exp(abs(100-(math.sqrt((x**2+y**2)/np.pi))))))+1)**0.1
    return a

def eggholder(x,y):
    a = -(y+47)*np.sin(math.sqrt(abs(x/2+(y+47))))-x*np.sin(math.sqrt(abs(x-(y+47))))
    return a

def holder_table(x,y):
    a = -abs(np.sin(x)*np.cos(y)*np.exp(abs(1-(math.sqrt(x**2+y**2))/np.pi)))
    return a

def mccormick(x,y):
    a = np.sin(x+y)+(x-y)**2-1.5*x+2.5*y+1
    return a

def schaffer_n2(x,y):
    a = 0.5 + (np.sin(x**2-y**2)**2-0.5)/(1+0.001*(x**2+y**2))**2
    return a

def schaffer_n4(x,y):
    a = 0.5 + (np.cos(np.sin(abs(x**2-y**2)))**2-0.5)/(1+0.001*(x**2+y**2))**2
    return a

#---------------------------------------------
func = {
        "ackley"          : ackley,
        "beale"           : beale,
        "goldstein_price" : goldstein_price,
        "booth"           : booth,
        "bukin"           : bukin,
        "matyas"          : matyas,
        "levi"            : levi,
        "himmelblau"      : himmelblau,
        "three_hump_camel": three_hump_camel,
        "easom"           : easom,
        "cross_in_tray"   : cross_in_tray,
        "eggholder"       : eggholder,
        "holder_table"    : holder_table,
        "mccormick"       : mccormick,
        "schaffer_n2"     : schaffer_n2,
        "schaffer_n4"     : schaffer_n4}

name = sys.argv[1]
f=func[name]

mt = meta.MT(f  = f,          # the function to be minimized
             n  = 1000,       # maximum number of iterations
             x0 = 4,          # initial x value
             y0 = 7,          # initial y value
             a  = 10,         # the neighborhood size
             T  = 100,        # initial temperature
             verbose = False) # whether to display details or not
# ev.set_verbose(True)
#---------------------------------------------
print("Random Search")
x,y = mt.random_search(-100,100)
print("x =",round(x,5),"\t","y =",round(y,5),"\t","f(x,y) =",round(f(x,y),5))
#---------------------------------------------
print("Simple Descent")
x,y = mt.simple_descent()
print("x =",round(x,5),"\t","y =",round(y,5),"\t","f(x,y) =",round(f(x,y),5))
#---------------------------------------------
print("Deepest Descent")
x,y = mt.deepest_descent()
print("x =",round(x,5),"  \t","y =",round(y,5),"\t","f(x,y) =",round(f(x,y),5))
#---------------------------------------------
print("Multistart Descent")
x,y = mt.multistart_descent()
print("x =",round(x,5),"\t","y =",round(y,5),"\t","f(x,y) =",round(f(x,y),5))
#---------------------------------------------
print("Tabu Search")
x,y = mt.tabu_search(100)
print("x =",round(x,5),"\t","y =",round(y,5),"\t","f(x,y) =",round(f(x,y),5))
#---------------------------------------------
print("Simulated Annealing")
x,y = mt.simulated_annealing(method="linear") # others: "exponential","discrete"
print("x =",round(x,5),"\t","y =",round(y,5),"\t","f(x,y) =",round(f(x,y),5))
#---------------------------------------------
print("Threshold Accept")
x,y = mt.threshold_accept(method="linear")    # others: "exponential","discrete"
print("x =",round(x,5),"\t","y =",round(y,5),"\t","f(x,y) =",round(f(x,y),5))
