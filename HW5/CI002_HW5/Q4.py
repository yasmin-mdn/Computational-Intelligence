# Q4_graded
# Do not change the above line.
# Here we use our unknown values as the genes.
# So our chromosome would be unknown values of the equation.
#f(x) = 168x^3 − 7.22x^2 + 15.5x − 13.2 = 0
# We consider our answer to be in range of [-9, 9]
# Ans we also accept 4 numbers of fraction.
import random
random.seed(45)
import operator
Coefficients = [168 , -7.22 , 15.5 , -13.2]
ChromosomeSize = 4


class Individual:
    def __init__(self):
        self.Chromosome = [None] * ChromosomeSize
        self.Fitness = 0
        self.Answer = 0
        self.Neg = random.randint(0, 1)

    def Initialize(self):
        String = ""
        for i in range(ChromosomeSize):
            if i is 1:
                self.Chromosome[i] = '.'
            else:
                self.Chromosome[i] = random.randint(0, 9)
            String += str(self.Chromosome[i])
        if self.Neg is 0:
            self.Answer = float(String)
        else:
            self.Answer = float(String) * -1

    def CalculateFitness(self):
        Value = 0
        for i in range(len(Coefficients)):
            Value += Coefficients[i] * pow(self.Answer, 5 - i)
        self.Fitness = abs(Value)

    def Set(self, Chro):
        self.Chromosome = Chro

    def SwapDot(self):
        DotInd = 0
        String = ""
        for i in range(ChromosomeSize):
            if self.Chromosome[i] is '.':
                DotInd = i
                break
        if DotInd is not 1:
            self.Chromosome[DotInd] = self.Chromosome[1]
            self.Chromosome[1] = '.'

        for i in range(ChromosomeSize):
            String += str(self.Chromosome[i])
        if self.Neg is 0:
            self.Answer = float(String)
        else:
            self.Answer = float(String) * -1


def SelectParents(Population, SizeOfSelection):
    Selected = []
    # Sort by their fitness.
    for i in range(SizeOfSelection):
        Selected.append(Population[i])
    return Selected


def GeneticChange(MutationRate, Best):
    # Crossover at first.
    NewPop = []

    # With some probability we have mutation.
    for i in range(len(Best) - 2):
        Child1 = CrossOverTwoPoint(Best[i], Best[i + 1])
        Child2 = CrossOverTwoPoint(Best[i], Best[i + 2])
        Prob = random.random() / 10
        if Prob < MutationRate:
            NewPop.append(Mutate(Child1))
            NewPop.append(Mutate(Child2))
        else:
            NewPop.append(Child1)
            NewPop.append(Child2)
    return NewPop


def Mutate(Parent):
    # Just swapping.
    Index1 = random.randint(0, len(Parent.Chromosome) - 1)
    Index2 = random.randint(0, len(Parent.Chromosome) - 1)
    while Index1 == 2 or Index2 == 2:
        if Index1 == 2:
            Index1 = random.randint(0, len(Parent.Chromosome) - 1)
        else:
            Index2 = random.randint(0, len(Parent.Chromosome) - 1)

    Temp = Parent.Chromosome[Index1]
    Parent.Chromosome[Index1] = Parent.Chromosome[Index2]
    Parent.Chromosome[Index2] = Temp
    Parent.SwapDot()
    Parent.CalculateFitness()
    return Parent


def CrossOverTwoPoint(Mom, Dad):
    # Two point.
    Child = [0] * ChromosomeSize
    Index1 = random.randint(0, len(Dad.Chromosome) - 1)
    Index2 = random.randint(0, len(Dad.Chromosome) - 1)

    if Index1 < Index2:
        for i in range(Index1, Index2):
            Child[i] = Dad.Chromosome[i]
        for i in range(0, Index1):
            for j in range(0, len(Mom.Chromosome)):
                if Mom.Chromosome[j] not in Child:
                    Child[i] = Mom.Chromosome[j]
                    break
        for i in range(Index2, len(Child)):
            for j in range(0, len(Mom.Chromosome)):
                if Mom.Chromosome[j] not in Child:
                    Child[i] = Mom.Chromosome[j]
                    break
    else:
        for i in range(Index2, Index1):
            Child[i] = Dad.Chromosome[i]
        for i in range(0, Index2):
            for j in range(0, len(Mom.Chromosome)):
                if Mom.Chromosome[j] not in Child:
                    Child[i] = Mom.Chromosome[j]
                    break
        for i in range(Index1, len(Child)):
            for j in range(0, len(Mom.Chromosome)):
                if Mom.Chromosome[j] not in Child:
                    Child[i] = Mom.Chromosome[j]
                    break

    ChildInd = Individual()
    ChildInd.Set(Child)
    ChildInd.SwapDot()
    ChildInd.CalculateFitness()

    return ChildInd

# Now it's time to create the initial population.
Population = []
InitialPopulation = 1000
UpperBound = 200
Iteration = 0
Mutation = 0.05
SizeOfSelection = 75

for i in range(InitialPopulation):
    Ind = Individual()
    Ind.Initialize()
    Ind.CalculateFitness()
    Population.append(Ind)

while Iteration < UpperBound:
    # Sort.
    Population.sort(key=operator.attrgetter('Fitness'), reverse=False)
    # Select individuals with better fitnesses.
    Best = SelectParents(Population, SizeOfSelection)

    # It supports both mutation and crossover.
    Population = GeneticChange(Mutation, Best)
    Iteration += 1

Answer = min(Population, key=operator.attrgetter('Fitness'))
print(Answer.Chromosome, Answer.Answer, "Fitness: ", Answer.Fitness)


