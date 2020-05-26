import random
from random import random, randint, seed
from copy import deepcopy
from statistics import mean

POPULATION_SIZE = 'population_size'
MAX_GENERATIONS = 'max_generations'
MAX_DEPTH = 'max_depth'
MIN_DEPTH = "min_depth"
TOURNAMENT_SIZE = 'tournament_size'
MUTATION_RATE = 'mutation_rate'
CROSSOVER_RATE = 'crossover_rate'

# TODO: Parameterized selection strategy ( tournament, rank, etc.. )

class GPF:
    """
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
    """
    def __init__(self,
                 dataset_function,
                 target_function,
                 fitness_function,
                 functions,
                 terminals,
                 parameters):
        self.dataset_function = dataset_function
        self.target_function = target_function
        self.fitness_function = fitness_function
        self.functions = functions
        self.terminals = terminals
        self.parameters = parameters

    def init_population(self):
        """
        Initialize the population based on the defined parameters using
        the ramp method
        """
        population_size = self.parameters[POPULATION_SIZE]
        max_depth = self.parameters[MAX_DEPTH]
        min_depth = self.parameters[MIN_DEPTH]

        group_size = int(population_size / ((max_depth+1)-(min_depth+1)) / 2)
        populations = []

        for depth in range(min_depth+1, max_depth+1):
            # grow
            for i in range(group_size):
                tree = GPTreeNode(self.functions,
                                  self.terminals,
                                  self.parameters)
                tree.random(grow=True, max_depth=depth)
                populations.append(tree)
            # full
            for i in range(group_size):
                tree = GPTreeNode(self.functions,
                                  self.terminals,
                                  self.parameters)
                tree.random(grow=False, max_depth=depth)
                populations.append(tree)

        return populations

    def fitness(self, node, dataset):
        """
        Fitness function
        """
        return self.fitness_function(node, dataset)

    def selection(self, population, fitnesses):
        """
        select one node using tournament selection
        """
        tournament_size = self.parameters[TOURNAMENT_SIZE]
        # Select contenders
        tournament = [randint(0, len(population) - 1) for i in range(tournament_size)]
        tournament_fitnesses = [fitnesses[tournament[i]] for i in range(tournament_size)]
        # Clone winner
        return deepcopy(population[tournament[tournament_fitnesses.index(max(tournament_fitnesses))]])

    def run(self):
        """
        Run the GP loop
        """
        seed(0) # Seed at 0 to reproduce the result for analysis

        max_generations = self.parameters[MAX_GENERATIONS]

        dataset = self.dataset_function()
        population = self.init_population()

        # length of population after rounding
        population_size = len(population)

        best_solution = None
        best_solution_fitness = 0
        best_solution_gen = 0

        fitnesses = [self.fitness_function(population[i], dataset)
                     for i in range(population_size)]

        self._print_header()

        for gen in range(max_generations):

            print("...", gen)

            next_population = []
            for i in range(population_size):
                parent1 = self.selection(population, fitnesses)
                parent2 = self.selection(population, fitnesses)
                parent1.crossover(parent2)
                parent1.mutate()
                next_population.append(parent1)

            population = next_population

            fitnesses = [self.fitness(population[i], dataset)
                         for i in range(population_size)]

            max_fitness = max(fitnesses)
            if max_fitness > best_solution_fitness:
                best_solution_fitness = max_fitness
                best_solution_gen = gen
                best_solution = deepcopy(population[fitnesses.index(max_fitness)])
                self._print_intermediary_summary(gen, max_fitness)

            if best_solution_fitness == 1:
                break

        self._print_result_summary(best_solution_fitness, best_solution_gen)

        return best_solution

    def _print_header(self):
        print("****************************************************************************")
        print(" GPF - START OF RUN")
        print("****************************************************************************")

    def _print_intermediary_summary(self, gen, max_fitness):
        print("*** Generation:", gen, ", Best solution fitness:", round(max_fitness, 3), "***")

    def _print_result_summary(self, best_solution_fitness, best_solution_gen):
        print("****************************************************************************")
        print(" END OF RUN")
        print("****************************************************************************")
        print("Generation: " + str(best_solution_gen))
        print("Fitness=", round(best_solution_fitness, 3))


class GPTreeNode:

    def __init__(self, functions,
                 terminals,
                 parameters,
                 data=None,
                 left=None,
                 right=None):
        self.functions = functions
        self.terminals = terminals
        self.parameters = parameters
        self.data = data
        self.left = left
        self.right = right

    def eval(self, x):
        """
        Eval the parse tree
        """
        if self.data in self.functions:
            return self.data(self.left.eval(x), self.right.eval(x))
        elif self.data == 'x':
            return x
        else:
            return self.data

    def label(self):
        """
        """
        return self.data.__name__ \
            if self.data in self.functions \
            else str(self.data)

    def random(self, grow, max_depth, depth=0):
        """
        Create a random subtree
        """
        func_len = len(self.functions) - 1
        term_len = len(self.terminals) - 1
        min_depth = self.parameters[MIN_DEPTH]

        if depth < min_depth or (depth < max_depth and not grow):
            self.data = self.functions[randint(0, func_len)]
        elif depth >= max_depth:
            self.data = self.terminals[randint(0, term_len)]
        # intermediate depth, grow
        else:
            if random() > 0.5:
                self.data = self.terminals[randint(0, term_len)]
            else:
                self.data = self.functions[randint(0, func_len)]

        if self.data in self.functions:
            self.left = GPTreeNode(self.functions,
                                   self.terminals,
                                   self.parameters)

            self.left.random(grow,
                             max_depth,
                             depth=depth + 1)

            self.right = GPTreeNode(self.functions,
                                    self.terminals,
                                    self.parameters)

            self.right.random(grow,
                              max_depth,
                              depth=depth + 1)

    def mutate(self):
        """
        Mutate function
        """
        mutation_rate = self.parameters[MUTATION_RATE]
        if random() < mutation_rate:
            self.random(grow=True, max_depth=2)
        elif self.left:
            self.left.mutate()
        elif self.right:
            self.right.mutate()

    def size(self):
        """
        Size of the tree node
        """
        if self.data in self.terminals:
            return 1
        left_value = self.left.size() if self.left else 0
        right_value = self.right.size() if self.right else 0
        return 1 + left_value + right_value

    def clone(self):
        """
        Clone (a subtree)
        """
        tree = GPTreeNode(self.functions, self.terminals, self.parameters)
        tree.data = self.data
        if self.left:
            tree.left = self.left.clone()
        if self.right:
            tree.right = self.right.clone()
        return tree

    # Refactor this to select and insert

    def select_node(self, count):
        """
        Select node at specific branch determined by count
        """
        count[0] -= 1
        if count[0] <= 1:
            return self.clone()
        else:
            ret = None
            if self.left and count[0] > 1:
                ret = self.left.select_node(count)
            if self.right and count[0] > 1:
                ret = self.right.select_node(count)
            return ret

    def insert_node(self, count, node):
        """
        Insert node at specific branch determined by count
        """
        count[0] -= 1
        if count[0] <= 1:
            self.data = node.data
            self.left = node.left
            self.right = node.right
        else:
            if self.left and count[0] > 1:
                self.left.insert_node(count, node)
            if self.right and count[0] > 1:
                self.right.insert_node(count, node)

    def crossover(self, other):
        """
        Cross over function
        """
        crossover_rate = self.parameters[CROSSOVER_RATE]
        if random() < crossover_rate:
            node_to_insert = other.select_node([randint(1, other.size())])
            self.insert_node([randint(1, self.size())], node_to_insert)

    def print(self, buffer, prefix, children_prefix):
        """
        Helper function to print a tree
        """
        buffer.append(prefix)
        buffer.append(self.label())
        buffer.append("\n")
        if self.left:
            self.left.print(buffer, children_prefix + "├── ", children_prefix + "│   ");
        if self.right:
            self.right.print(buffer, children_prefix + "└── ", children_prefix + "    ");

    def __str__(self):
        buffer = []
        self.print(buffer, "", "")
        return "".join(buffer)


def main():
    parameters = {POPULATION_SIZE: 90,
                  MIN_DEPTH: 2,
                  MAX_DEPTH: 5,
                  MAX_GENERATIONS: 100,
                  TOURNAMENT_SIZE: 5,
                  MUTATION_RATE: 0.2,
                  CROSSOVER_RATE: 0.8 }

    def add(x, y): return x + y
    def sub(x, y): return x - y
    def mul(x, y): return x * y

    funcs = [add, sub, mul]
    terms = ['x', 0, 1, 2]

    def quadratic_func(x):
        """
        Quadratic function x^3 + x^2 + x^1 + 1
        """
        return x * x * x + x * x + x + 1

    def dataset_func():
        """
        Generate data points over the range 0 to 100 with i=2
        """
        return [[x/100, quadratic_func(x/100)] for x in range(0, 100, 2)]

    def NMAE(node, dataset):
        """
        mean absolute error over dataset normalized to [0,1]
        """
        return 1 / (1 + mean([abs(node.eval(ds[0]) - ds[1]) for ds in dataset]))

    gpf = GPF(dataset_function=dataset_func,
              target_function=quadratic_func,
              fitness_function=NMAE,
              functions=funcs,
              terminals=terms,
              parameters=parameters)

    best_solution = gpf.run()

    print(best_solution)
    print("best_solution.eval(2) == target_func(2) ==", best_solution.eval(2),
          "=", best_solution.eval(2) == quadratic_func(2))

if __name__ == "__main__":
    main()
