# Week 1

Run the following commands.
```shell
pip install -r requirements.txt
pip install -e .
```

# Solvers
Pyomo internally uses one of off-the-shelf optimization solvers to solve an optimization problem.
For the demos that we have here, you could use solvers like `glpk` and `cbc`. 
Both of them are able to solve linear programming problem.

## GLPK
If you are using Conda, you should be able to install the solver using:
```shell
conda install glpk
```
See https://github.com/conda-forge/glpk-feedstock for more info.

## Cbc
### Mac users
You can install the `cbc` solver with 
```shell
brew install cbc
```

### Windows users
You could install a binary as this instruction says: https://github.com/coin-or/Cbc
