# Makefile.local must define the following variables
#   LIBDIR      install dir for C++ libraries
#   INCDIR      install dir for C++ headers
#   CPP         C++ compiler command line


# Note that INCFILES are in the rf_kernels/ subdirectory
INCFILES = online_mask_filler.hpp xorshift_plus.hpp 

OFILES = online_mask_filler.o

TESTBINFILES = test-online-mask-filler

UNITTEST_TOUCHFILES=$(addprefix unittest_touchfiles/ut_,$(TESTBINFILES))


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

test: $(TESTBINFILES) $(UNITTEST_TOUCHFILES)

install: librf_kernels.so
	mkdir -p $(INCDIR)/rf_kernels $(LIBDIR)/
	for f in $(INCFILES); do cp rf_kernels/$$f $(INCDIR)/rf_kernels; done
	cp -f librf_kernels.so $(LIBDIR)/
	cp -f rf_kernels.hpp $(INCDIR)/

uninstall:
	rm -f $(LIBDIR)/librf_kernels.so
	rm -f $(INCDIR)/rf_kernels/*.hpp $(INCDIR)/rf_kernels.hpp
	rmdir $(INCDIR)/rf_kernels

clean:
	rm -f $(TESTBINFILES) *~ *.o *.so *.pyc rf_kernels/*~ unittest_touchfiles/ut_*
	if [ -d unittest_touchfiles ]; then rmdir unittest_touchfiles; fi

unittest_touchfiles/ut_%: %
	mkdir -p unittest_touchfiles && ./$< && touch $@


####################################################################################################


online_mask_filler.o: online_mask_filler.cpp rf_kernels/xorshift_plus.hpp rf_kernels/online_mask_filler.hpp
	$(CPP) -c -o $@ $<

unit_testing.o: unit_testing.cpp rf_kernels/unit_testing.hpp
	$(CPP) -c -o $@ $<

test-online-mask-filler.o: test-online-mask-filler.cpp rf_kernels/xorshift_plus.hpp rf_kernels/online_mask_filler.hpp
	$(CPP) -c -o $@ $<

librf_kernels.so: $(OFILES)
	$(CPP) $(CPP_LFLAGS) -shared -o $@ $^

test-online-mask-filler: test-online-mask-filler.o online_mask_filler.o
	$(CPP) $(CPP_LFLAGS) -o $@ $^
