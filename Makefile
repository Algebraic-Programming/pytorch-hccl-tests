.PHONY: clean clean-build clean-pyc clean-test coverage dist docs help install lint lint/flake8 lint/black
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

lint/flake8: ## check style with flake8
	flake8 pytorch_hccl_tests tests
lint/black: ## check style with black
	black --check pytorch_hccl_tests tests

lint: lint/flake8 lint/black ## check style

test: ## run tests quickly with the default Python
	pytest

test-all: ## run tests on every Python version with tox
	tox

coverage: ## check code coverage quickly with the default Python
	coverage run --source pytorch_hccl_tests -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/pytorch_hccl_tests.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ pytorch_hccl_tests
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

release: dist ## package and upload a release
	twine upload dist/*

dist: clean ## builds source and wheel package
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

install: clean ## install the package to the active Python's site-packages
	pip install . -f https://download.pytorch.org/whl/torch_stable.html


DEVICE = cpu

latency: ## OSU MPI/HCCL latency benchmark
	export OMP_NUM_THREADS=1
	torchrun --nnodes 1 --nproc_per_node 2 pytorch_hccl_tests/osu/p2p/osu_latency.py --device ${DEVICE}

multi-latency: ## OSU MPI/HCCL multi-latency benchmark
	export OMP_NUM_THREADS=1
	torchrun --nnodes 1 --nproc_per_node 4 pytorch_hccl_tests/osu/p2p/osu_multi_lat.py --device ${DEVICE}


bandwidth: ## OSU MPI/HCCL bandwidth benchmark
	export OMP_NUM_THREADS=1
	torchrun --nnodes 1 --nproc_per_node 2 pytorch_hccl_tests/osu/p2p/osu_bw.py --device ${DEVICE}

bidirectional-bw: ## OSU MPI/HCCL bidirectional bandwidth benchmark
	export OMP_NUM_THREADS=1
	torchrun --nnodes 1 --nproc_per_node 2 pytorch_hccl_tests/osu/p2p/osu_bibw.py --device ${DEVICE}

allreduce: ## OSU MPI/HCCL allreduce benchmark
	export OMP_NUM_THREADS=1
	torchrun --nnodes 1 --nproc_per_node 4 pytorch_hccl_tests/osu/collectives/osu_allreduce.py --device ${DEVICE}
	torchrun --nnodes 1 --nproc_per_node 4 pytorch_hccl_tests/osu/collectives/osu_allgather.py --device ${DEVICE}
	torchrun --nnodes 1 --nproc_per_node 4 pytorch_hccl_tests/osu/collectives/osu_alltoall.py --device ${DEVICE}

allgather: ## OSU MPI/HCCL allgather benchmark
	export OMP_NUM_THREADS=1
	torchrun --nnodes 1 --nproc_per_node 4 pytorch_hccl_tests/osu/collectives/osu_allgather.py --device ${DEVICE}

alltoall: ## OSU MPI/HCCL alltoall benchmark
	export OMP_NUM_THREADS=1
	torchrun --nnodes 1 --nproc_per_node 4 pytorch_hccl_tests/osu/collectives/osu_alltoall.py --device ${DEVICE}

barrier: ## OSU MPI/HCCL barrier benchmark
	export OMP_NUM_THREADS=1
	torchrun --nnodes 1 --nproc_per_node 2 pytorch_hccl_tests/osu/collectives/osu_barrier.py --device ${DEVICE}

broadcast: ## OSU MPI/HCCL broadcast benchmark
	export OMP_NUM_THREADS=1
	torchrun --nnodes 1 --nproc_per_node 4 pytorch_hccl_tests/osu/collectives/osu_broadcast.py --device ${DEVICE}

gather: ## OSU MPI/HCCL Bandwidth benchmark
	export OMP_NUM_THREADS=1
	torchrun --nnodes 1 --nproc_per_node 4 pytorch_hccl_tests/osu/collectives/osu_gather.py --device ${DEVICE}

reduce: ## OSU MPI/HCCL Bandwidth benchmark
	export OMP_NUM_THREADS=1
	torchrun --nnodes 1 --nproc_per_node 4 pytorch_hccl_tests/osu/collectives/osu_reduce.py --device ${DEVICE}

scatter: ## OSU MPI/HCCL Bandwidth benchmark
	export OMP_NUM_THREADS=1
	torchrun --nnodes 1 --nproc_per_node 2 pytorch_hccl_tests/osu/collectives/osu_scatter.py --device ${DEVICE}

p2p: ## OSU MPI/HCCL point-to-point benchmark suite
	@export OMP_NUM_THREADS=1
	torchrun --nnodes 1 --nproc_per_node 2 pytorch_hccl_tests/osu/p2p/osu_latency.py --device ${DEVICE}
	torchrun --nnodes 1 --nproc_per_node 2 pytorch_hccl_tests/osu/p2p/osu_bw.py --device ${DEVICE}
	torchrun --nnodes 1 --nproc_per_node 2 pytorch_hccl_tests/osu/p2p/osu_bibw.py --device ${DEVICE}
	torchrun --nnodes 1 --nproc_per_node 2 pytorch_hccl_tests/osu/p2p/osu_multi_lat.py --device ${DEVICE}

collectives: ## OSU MPI/HCCL collective communications benchmark suite
	@export OMP_NUM_THREADS=1
	torchrun --nnodes 1 --nproc_per_node 4 pytorch_hccl_tests/osu/collectives/osu_allreduce.py --device ${DEVICE}
	torchrun --nnodes 1 --nproc_per_node 4 pytorch_hccl_tests/osu/collectives/osu_allgather.py --device ${DEVICE}
	torchrun --nnodes 1 --nproc_per_node 4 pytorch_hccl_tests/osu/collectives/osu_alltoall.py --device ${DEVICE}
	torchrun --nnodes 1 --nproc_per_node 2 pytorch_hccl_tests/osu/collectives/osu_barrier.py --device ${DEVICE}
	torchrun --nnodes 1 --nproc_per_node 4 pytorch_hccl_tests/osu/collectives/osu_broadcast.py --device ${DEVICE}
	torchrun --nnodes 1 --nproc_per_node 4 pytorch_hccl_tests/osu/collectives/osu_gather.py --device ${DEVICE}
	torchrun --nnodes 1 --nproc_per_node 4 pytorch_hccl_tests/osu/collectives/osu_reduce.py --device ${DEVICE}
	torchrun --nnodes 1 --nproc_per_node 2 pytorch_hccl_tests/osu/collectives/osu_scatter.py --device ${DEVICE}
	

benchmarks: p2p collectives ## OSU MPI/HCCL complete benchmark suite
	