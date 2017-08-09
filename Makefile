# Makefile.local must define the following variables
#   LIBDIR      install dir for C++ libraries
#   INCDIR      install dir for C++ headers
#   CPP         C++ compiler command line


# Note that INCFILES are in the rf_kernels/ subdirectory
INCFILES = xorshift_plus.hpp 

OFILES = online_mask_filler.o

TESTBINFILES = test-online-mask-filler


####################################################################################################


include Makefile.local

ifndef CPP
$(error Fatal: Makefile.local must define CPP variable)
endif

ifndef INCDIR
$(error Fatal: Makefile.local must define INCDIR variable)
endif

ifndef LIBDIR
$(error Fatal: Makefile.local must define LIBDIR variable)
endif


all: librf_kernels.so $(TESTBINFILES)

install: librf_kernels.so
	mkdir -p $(INCDIR)/rf_kernels $(LIBDIR)/
	cp -f rf_kernels.hpp $(INCDIR)/
	cp -f $(INCFILES) $(INCDIR)/rf_kernels
	cp -f librf_kernels.so $(LIBDIR)/

uninstall:
	rm -f $(LIBDIR)/librf_kernels.so
	rm -f $(INCDIR)/rf_kernels/*.hpp $(INCDIR)/rf_kernels.hpp
	rmdir $(INCDIR)/rf_kernels

clean:
	rm -f $(TESTBINFILES) *~ *.o *.so *.pyc rf_kernels/*.hpp


####################################################################################################


online_mask_filler.o: online_mask_filler.cpp rf_kernels/xorshift_plus.hpp rf_kernels/online_mask_filler.hpp
	$(CPP) -c -o $@ $<

test-online-mask-filler.o: test-online-mask-filler.cpp rf_kernels/xorshift_plus.hpp rf_kernels/online_mask_filler.hpp
	$(CPP) -c -o $@ $<

librf_kernels.so: $(OFILES)
	$(CPP) $(CPP_LFLAGS) -shared -o $@ $^

test-online-mask-filler: test-online-mask-filler.o online_mask_filler.o
	$(CPP) $(CPP_LFLAGS) -o $@ $^
