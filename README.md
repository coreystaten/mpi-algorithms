# MPI Algorithm Exploration

This repo contains simplified implementations of various MPI algorithms in Python, along with visualizations and a simulated network timer.  These implementations are for educational purposes only.

This code isn't designed for production use.  The provided `Communicator` is designed to work with multiple Python threads in a single process.


## Why?
I wanted to understand the behavior and performance characteristics of collective communication primitives.  I knew a mishmash of facts ("ring allgather is faster than gather+broadcast"), but had no cohesive picture.

Usually, I would read the original source code to understand better -- but the underlying code was in C and had a lot of low-level details around networking and buffer management mixed in with the core algorithmic ideas, so it was kind of a pain. Python is a lot closer to executable pseudocode than C is, and writing code is one of the best ways to learn something, so I made these toy Python implementations.


## How to Use

Browse the implementations in `mpi_algorithms/algorithms`. You can see generated diagrams showing the network send/recv topology of the algorithms in `diagrams`. The scaling of individual diagrams as world size increases is available in `charts`. Note that some primitives have constant total data size as world size increases (such as broadcast), whereas others have constant per-rank data size (like allreduce).

To re-generate charts and diagrams:
- Run `make env` to create a virtual environment and install dependencies, then `source env/bin/activate`.
- Run the script `python scripts/run_all`.  

To add a new algorithm:
- Add it to the appropriate file in `mpi_algorithms/algorithms`
- Add expected behavior to `primitive_cases` in `test/tests.py`. Run with `make test`.
- Add a run case to the primitive list in `scripts/run_all.py`. Rerun `scripts/run_all.py` to generate the new charts and diagrams. This can take a while, as there's currently no caching.  The simulated latency and throughput arguments can also be changed when running `scripts/run_all.py` to generate diagrams for different network environments.

Pull requests for additional algorithms welcome.

