# Makefile.local must define the following variables
#   LIBDIR      install dir for C++ libraries
#   INCDIR      install dir for C++ headers
#   CPP         C++ compiler command line


INCFILES = rf_kernels.hpp

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
	mkdir -p $(INCDIR)/ $(LIBDIR)/
	cp -f $(INCFILES) $(INCDIR)/
	cp -f librf_kernels.so $(LIBDIR)/

uninstall:
	for f in $(INCFILES); do rm -f $(INCDIR)/$$f; done
	rm -f $(LIBDIR)/librf_kernels.so

clean:
	rm -f $(TESTBINFILES) *~ *.o *.so *.pyc


####################################################################################################


%.o: %.cpp $(INCFILES)
	$(CPP) -c -o $@ $<

librf_kernels.so: $(OFILES)
	$(CPP) $(CPP_LFLAGS) -shared -o $@ $^

test-online-mask-filler: test-online-mask-filler.cpp $(INCFILES) librf_kernels.so
	$(CPP) $(CPP_LFLAGS) -o $@ $< -lrf_kernels $(LIBS)
