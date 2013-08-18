ROOT=..

all: *.py Makefile
	mkdir -p $(ROOT)/build/nnkit/
	touch $(ROOT)/build/__init__.py
	cp $(ROOT)/src/*.py $(ROOT)/build/nnkit/
	(cd examples/; make)

clean:
	rm -rf $(ROOT)/build/nnkit/
	rm -f $(ROOT)/build/__init__.py

.PHONY: all clean
