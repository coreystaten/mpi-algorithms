# MPI Algorithm Exploration

This repo contains simplified implementations of various MPI algorithms in Python, along with visualizations and simulation-based speed estimates.  These implementations are for educational purposes only.


## Why?
I wanted to understand the behavior and performance characteristics of collective communication primitives.  I knew a mishmash of facts ("ring allgather is faster than gather+broadcast"), but had no cohesive picture.

I would usually read the original source code to develop understanding -- but the [the code](https://github.com/NVIDIA/nccl) was in C and had a lot of low-level details around networking and buffer management mixed in with the core algorithmic ideas, so it was kind of a pain. Python is a lot closer to executable pseudocode than C is.


## How to Use

Browse the implementations in `mpi_algorithms/algorithms`. You can view diagrams showing the network send/recv topology of the algorithms in `diagrams`. The simulated speed of individual algorithms as world size increases is available in `charts`.

To re-generate charts and diagrams:
- Run `make env` to create a virtual environment and install dependencies, then `source env/bin/activate`.
- Run `python scripts/run_all`.  

To add a new algorithm:
- Add it to the appropriate file in `mpi_algorithms/algorithms`
- Add expected behavior to `primitive_cases` in `test/tests.py`. Check with `make test`.
- Add a run case to the primitive list in `scripts/run_all.py`. Rerun `scripts/run_all.py` to generate the new charts and diagrams. The simulated latency and throughput arguments can also be changed when running `scripts/run_all.py` to generate scaling diagrams for different network environments.

Pull requests for additional algorithms welcome.

