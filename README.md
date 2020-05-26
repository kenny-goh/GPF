GPF is a simple tree-based genetic programming framework that enables various GP problems
to be solved generically using parameterized dataset function, target function, fitness function, functions,
terminals and GP parameters.

Let's use a minimal example to solve a symbolic regression problem:

def add(x, y): return x + y
def sub(x, y): return x - y
def mul(x, y): return x * y

funcs = [add, sub, mul]
terms = ['x', 0, 1, 2]

def quadratic_func(x): return x * x * x + x * x + x + 1
def dataset_func(): return [[x/100, target_func(x/100)] for x in range(0, 100, 2)]
def NMAE(node, dataset): return 1 / (1 + mean([abs(node.eval(ds[0]) - ds[1]) for ds in dataset]))

parameters = { POPULATION_SIZE: 90,
               MIN_DEPTH: 2,
               MAX_DEPTH: 5,
               MAX_GENERATIONS: 100,
               TOURNAMENT_SIZE: 5,
               MUTATION_RATE: 0.2,
               CROSSOVER_RATE: 0.8 }

gpf = GPF( dataset_function=quadratic_func,
           target_function=target_func,
           fitness_function=NMAE,
           functions=funcs,
           terminals=terms,
           parameters=parameters)

gpf.run()

```
Resulting program:
    add
    ├── add
    │   ├── x
    │   └── add
    │       ├── x
    │       └── mul
    │           ├── add
    │           │   ├── mul
    │           │   │   ├── add
    │           │   │   │   ├── mul
    │           │   │   │   │   ├── x
    │           │   │   │   │   └── x
    │           │   │   │   └── sub
    │           │   │   │       ├── x
    │           │   │   │       └── 1
    │           │   │   └── add
    │           │   │       ├── 0
    │           │   │       └── x
    │           │   └── add
    │           │       ├── 1
    │           │       └── 0
    │           └── mul
    │               ├── 1
    │               └── 1
    └── mul
        ├── sub
        │   ├── 1
        │   └── 1
        └── 1
 ```
 