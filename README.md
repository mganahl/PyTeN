
# PyTeN
PyTeN is a small python-only tensor network library that can be used to calculate 
ground-states and do real time evolution of 1d quantum systems. 


## finite systems:

* ground states using DMRG
* Real time evolution using TEBD or TDVP
* supports MPO encoding of Hamiltonians

## infinite systems

* ground states for translationally invariant systems using gradient optimization
* ground states for systems with finite unit-cells using IDMRG

the examples folder contains examples of how to use it

Dependencies:

* scipy and numpy installation
* ncon (https://github.com/mhauru/ncon) (ships with this repository)

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
### Prerequisites

You need an numpy and scipy installation. To run the test cases, you additionally need the cython compiler installed on your OS (see below)

### Installing
To generate the binary files needed for testing, cd to 
```
cd /PyTeN/tests/HeisED
```
and run 
```
python setup.py build_ext --inplace
```
This should create a file XXZED.so (or a similar name, depending on the installed c-compiler). 


## Running the tests
Once the .so file from above is generated, run
```
python testCasesMPS.py
python DMRGtest.py
python timeevtests.py
```
which should all finish without error (some warnings may show up).

## Authors

* **Martin Ganahl** 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Ash Milsted (https://github.com/amilsted/evoMPS)
* Markus Hauru (https://github.com/mhauru/ncon)

