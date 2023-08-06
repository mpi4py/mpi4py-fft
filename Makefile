VERSION=$(shell python3 -c "import mpi4py_fft; print(mpi4py_fft.__version__)")

default:
	python setup.py build build_ext -i

pip:
	rm -f dist/*
	python setup.py sdist
	twine upload dist/*

tag:
	git tag $(VERSION)
	git push --tags

publish: tag pip

clean:
	git clean -dxf mpi4py_fft
	cd docs && make clean && cd ..
	@rm -rf *.egg-info/ build/ dist/ .eggs/
