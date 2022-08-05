# large-scale-truss-optimization
# Solid rocket motor design using multi-objective optimization
Multi-objective optimization of the weight and compliance of a large 3D truss subject to 
stress and displacement constraints [1] using NSGA-II [3]. Pymoo [2] is used for 
implementing NSGA-II. Finite element analysis (FEA) is performed in order to calculate 
the truss compliance. The FEA code is available under both Python and MATLAB.

<h2>Running unit tests to ensure everything is in order:</h4>
1. Clone the repo:

    ```https://github.com/abhiroopghosh71/large-scale-truss-optimization.git```
    
2. Change into the working directory ```cd large-scale-truss-optimization```

3. To install the dependencies use ```pip install requirements.txt```.
If you are using Anaconda, it is recommended to create a new virtual environment 
using ```conda create --name <envname> python=3.8 --file requirements.txt```

4. OPTIONAL: Install the MATLAB Engine API for Python if the MATLAB FEA code is preferred. 
Instructions can he found on the [MathWorks website](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)

5. Change into the tests directory ```cd tests```

6. To test the Python codes run: ```pytest test_truss_python_fea.py``` 
and ```pytest test_truss_pymoo_python_fea_parallel.py```

7. If MATLAB Engine API was installed previously, run the 
tests: ```pytest test_truss_matlab_fea.py``` 
and ```pytest test_truss_pymoo_matlab_fea.py```

<h2>Running the optimization</h4>

1. Change into the ```large-scale-truss-optimization``` directory.

2. To run the optimization use: ```python optimize.py [OPTIONS]```. 
For example, to run the optimization with a 40 population size and 100 generations, run: 
```python optimize.py --popsize 40 --ngen 100```

<h2>Important command line optimization parameters</h4>

1. ```--seed <value>```: Sets the seed for the random number generator.

2. ```--ngen <value>```: Number of generations of NSGA-II.

3. ```--popsize <value>```: Population size of NSGA-II.
4. ```--nshapevar <value>```: Number of shape variables [1].
5. ```--symmetric```: Enforcing symmetry in the truss [1].

Please report issues to me, Abhiroop Ghosh, at ghoshab1@msu.edu.

<h2>References:</h4>
1. A. Ghosh, K. Deb, E. Goodman, R. Averill, A. Diaz, "Combining User Knowledge and Online Innovization for Faster Solution to Multi-objective Design Optimization Problems," Combining User Knowledge and Online Innovization for Faster Solution to Multi-objective Design Optimization Problems, https://doi.org/10.1007/978-3-030-72062-9_9

2. J. Blank and K. Deb, pymoo: Multi-Objective Optimization in Python, in IEEE Access, vol. 8, pp. 89497-89509, 2020, https://doi.org/10.1109/ACCESS.2020.2990567

3. K. Deb, A. Pratap, S. Agarwal and T. Meyarivan, "A fast and elitist multiobjective genetic algorithm: NSGA-II," in _IEEE Transactions on Evolutionary Computation_, vol. 6, no. 2, pp. 182-197, April 2002. https://doi.org/10.1109/4235.996017

