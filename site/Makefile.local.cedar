# Makefile.local for cedar.computecanada.ca

# Directory where C++ libraries will be installed
LIBDIR=$(HOME)/lib

# Directory where C++ header files will be installed
INCDIR=$(HOME)/include

CPP=g++ -std=c++11 -pthread -fPIC -march=native -Wall -O3 -ffast-math -funroll-loops --param inline-unit-growth=10000 -I. -I$(INCDIR)
CPP_LFLAGS= -L. -L$(LIBDIR)
