## 3D Metaheuristics in Python
The implemented 3D metaheuristics are:

* Random Search (RS),
* Simple Descent (SD),
* Deepest Descent (DS),
* Multistart Descent (MD),
* Tabu Search (TS),
* Simulated Annealing (SA),
* Threshold Accept (TA).

All of the implemented algorithms can be used to find the minimum of 3D function.  
For example : f(x,y) = x^2+y^2-4

### Requirement
```
python==3.7.0
numpy==1.18.1
```
### How to use

Open test.py you will find some examples
```
import src.mt as meta
import numpy as np

def ackley(x,y):
    a = -20*np.exp(-0.2*math.sqrt(0.5*(x**2+y**2)))\
        -np.exp(0.5*(np.cos(2*np.pi*x)+(np.cos(2*np.pi*y))))+np.e+20
    return a

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

mt = meta.MT(f=f,n=1000,x0=4,y0=7,a=10,T=100,verbose=False)

x,y = mt.random_search(-100,100)
print("x =",round(x,5),"\t","y =",round(y,5),"\t","f(x,y) =",round(f(x,y),5))
```
