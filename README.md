
# PyTeN
PyTeN is a small python3-only tensor network library that can be used to calculate 
ground-states and do real time evolution of 1d quantum systems. 


## finite systems:

* ground states using DMRG
* Real time evolution using TEBD or TDVP
* supports MPO encoding of Hamiltonians

  **TDVP does currently not support dynamical expansion of the bond dimension**


## infinite systems

* ground states for translational invariant systems using gradient optimization (https://arxiv.org/abs/1801.02219)
* ground states for translational invariant systems using VUMPS (https://arxiv.org/pdf/1701.07035.pdf)
* ground states for systems with finite unit-cells using IDMRG


the examples folder contains examples of how to use it

Dependencies:

* scipy and numpy installation
* ncon (https://github.com/mhauru/ncon) (ships with this repository)
* matplotlib

## Getting Started
Clone the repository and cd to the examples folder:

```
cd PyTeN/examples
```
and run the example scripts. 
HeisMPS.py runs DMRG for a Heisenberg model:
```
python HeisMPS.py
```
TEBD.py and TDVP.py calculate the ground state of a N site Heisenberg model, then applies an S+ at the center
and evolves the state using TEBD (https://arxiv.org/abs/quant-ph/0301063) or TDVP (https://arxiv.org/abs/1408.5056) algorithm. Run 
```
python TEBD.py
python TDVP.py
```
to do the evolution.

The above examples-scripts are commented to explain the individual steps and help you to get started with new models.
### Getting help
There is currently no documentation, but most methods have docstring explaining what they do.

### Prerequisites

You need an numpy and scipy installation. I have tested the code with numpy 1.14.3 and scipy 0.19.0 and 1.1.0.
To run the test cases, you additionally need the cython compiler installed on your OS (see below)

### Installing
To generate the binary files needed for testing, cd to 
```
cd PyTeN/tests/HeisED
```
and run 
```
python setup.py build_ext --inplace
```
This should create a file XXZED.so (or a similar name, depending on the installed c compiler. gcc might
throw some warnings at you).


## Running the tests
Once the .so file from above is generated, run
```
python testCasesMPS.py
python DMRGtest.py
python timeevtests.py
```
which should all finish without error (some warnings may show up).

## Extending
New models can be implemented using the MPO formalism. In lib/mpslib/Hamiltonians.py you can find a few examples of how
this is done. Any new Hamiltonian should be implemented by declaring a new class. The class should be derived from the
base class MPO (see top of Hamiltonians.py). I use the following convention for building nearest neighbor MPOs:
```
11
O_11
.
.
.
O_N1, O_N2, ..., O_NN, 11
```
11 and O_ij are local physical operators appearing in the Hamiltonian. Some functions assume that the MPO has such a form (for example the member MPO.twoSiteGate(m,n,tau) for obtaining unitary gates for TEBD), so I strongly advise to conform to this convention.

## Authors

* **Martin Ganahl** 
If you have any question, suggestions or concerns feel free to drop me an email or use the Issues system.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Ash Milsted (https://github.com/amilsted/evoMPS)
* Markus Hauru (https://github.com/mhauru/ncon)

