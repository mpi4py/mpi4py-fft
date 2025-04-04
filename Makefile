VERSION=$(shell python -c "import mpi4py_fft; print(mpi4py_fft.__version__)")

default:
	python setup.py build build_ext -i

tag:
	git tag $(VERSION)
	git push --tags

clean:
	git clean -dxf mpi4py_fft
	cd docs && make clean && cd ..
	@rm -rf *.egg-info/ build/ dist/ .eggs/
