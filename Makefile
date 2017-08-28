# Makefile.local must define the following variables
#   LIBDIR      install dir for C++ libraries
#   INCDIR      install dir for C++ headers
#   CPP         C++ compiler command line


# Note that INCFILES are in the rf_kernels/ subdirectory
INCFILES = \
  core.hpp \
  internals.hpp \
  downsample.hpp \
  downsample_internals.hpp \
  online_mask_filler.hpp \
  polynomial_detrender.hpp \
  polynomial_detrender_internals.hpp \
  spline_detrender.hpp \
  spline_detrender_internals.hpp \
  unit_testing.hpp \
  upsample.hpp \
  upsample_internals.hpp \
  xorshift_plus.hpp 

OFILES = \
  downsample.o \
  misc.o \
  online_mask_filler.o \
  polynomial_detrender.o \
  spline_detrender.o \
  upsample.o

TESTBINFILES = \
  test-downsample \
  test-online-mask-filler \
  test-spline-detrender \
  test-upsample

TIMEBINFILES = \
  time-memory-access-patterns \
  time-online-mask-filler \
  time-polynomial-detrender \
  time-spline-detrender \
  time-upsample

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


all: librf_kernels.so $(TESTBINFILES) $(TIMEBINFILES)

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


downsample.o: downsample.cpp rf_kernels/internals.hpp rf_kernels/downsample.hpp rf_kernels/downsample_internals.hpp
	$(CPP) -c -o $@ $<

misc.o: misc.cpp rf_kernels/core.hpp
	$(CPP) -c -o $@ $<

online_mask_filler.o: online_mask_filler.cpp rf_kernels/internals.hpp rf_kernels/xorshift_plus.hpp rf_kernels/online_mask_filler.hpp
	$(CPP) -c -o $@ $<

polynomial_detrender.o: polynomial_detrender.cpp rf_kernels/core.hpp rf_kernels/internals.hpp rf_kernels/polynomial_detrender.hpp rf_kernels/polynomial_detrender_internals.hpp
	$(CPP) -c -o $@ $<

spline_detrender.o: spline_detrender.cpp rf_kernels/internals.hpp rf_kernels/spline_detrender.hpp rf_kernels/spline_detrender_internals.hpp
	$(CPP) -c -o $@ $<

unit_testing.o: unit_testing.cpp rf_kernels/internals.hpp rf_kernels/unit_testing.hpp
	$(CPP) -c -o $@ $<

upsample.o: upsample.cpp rf_kernels/internals.hpp rf_kernels/upsample.hpp rf_kernels/upsample_internals.hpp
	$(CPP) -c -o $@ $<


test-downsample.o: test-downsample.cpp rf_kernels/internals.hpp rf_kernels/unit_testing.hpp rf_kernels/downsample.hpp rf_kernels/downsample_internals.hpp
	$(CPP) -c -o $@ $<

test-online-mask-filler.o: test-online-mask-filler.cpp rf_kernels/internals.hpp rf_kernels/unit_testing.hpp rf_kernels/xorshift_plus.hpp rf_kernels/online_mask_filler.hpp
	$(CPP) -c -o $@ $<

test-spline-detrender.o: test-spline-detrender.cpp rf_kernels/internals.hpp rf_kernels/unit_testing.hpp rf_kernels/spline_detrender.hpp rf_kernels/spline_detrender_internals.hpp
	$(CPP) -c -o $@ $<

test-upsample.o: test-upsample.cpp rf_kernels/internals.hpp rf_kernels/unit_testing.hpp rf_kernels/upsample.hpp rf_kernels/upsample_internals.hpp
	$(CPP) -c -o $@ $<


#time-downsample.o: time-downsample.cpp rf_kernels/internals.hpp rf_kernels/unit_testing.hpp rf_kernels/downsample.hpp rf_kernels/downsample_internals.hpp
#	$(CPP) -c -o $@ $<

time-memory-access-patterns.o: time-memory-access-patterns.cpp rf_kernels/internals.hpp rf_kernels/unit_testing.hpp
	$(CPP) -c -o $@ $<

time-online-mask-filler.o: time-online-mask-filler.cpp rf_kernels/internals.hpp rf_kernels/unit_testing.hpp rf_kernels/xorshift_plus.hpp rf_kernels/online_mask_filler.hpp
	$(CPP) -c -o $@ $<

time-polynomial-detrender.o: time-polynomial-detrender.cpp rf_kernels/core.hpp rf_kernels/internals.hpp rf_kernels/unit_testing.hpp rf_kernels/polynomial_detrender.hpp rf_kernels/polynomial_detrender_internals.hpp
	$(CPP) -c -o $@ $<

time-spline-detrender.o: time-spline-detrender.cpp rf_kernels/internals.hpp rf_kernels/unit_testing.hpp rf_kernels/spline_detrender.hpp rf_kernels/spline_detrender_internals.hpp
	$(CPP) -c -o $@ $<

time-upsample.o: time-upsample.cpp rf_kernels/internals.hpp rf_kernels/unit_testing.hpp rf_kernels/upsample.hpp rf_kernels/upsample_internals.hpp
	$(CPP) -c -o $@ $<


librf_kernels.so: $(OFILES)
	$(CPP) $(CPP_LFLAGS) -shared -o $@ $^


test-downsample: test-downsample.o downsample.o
	$(CPP) $(CPP_LFLAGS) -o $@ $^

test-online-mask-filler: test-online-mask-filler.o online_mask_filler.o
	$(CPP) $(CPP_LFLAGS) -o $@ $^

test-spline-detrender: test-spline-detrender.o spline_detrender.o
	$(CPP) $(CPP_LFLAGS) -o $@ $^

test-upsample: test-upsample.o upsample.o
	$(CPP) $(CPP_LFLAGS) -o $@ $^


time-memory-access-patterns: time-memory-access-patterns.o unit_testing.o
	$(CPP) $(CPP_LFLAGS) -o $@ $^

time-online-mask-filler: time-online-mask-filler.o online_mask_filler.o unit_testing.o
	$(CPP) $(CPP_LFLAGS) -o $@ $^

time-spline-detrender: time-spline-detrender.o unit_testing.o spline_detrender.o
	$(CPP) $(CPP_LFLAGS) -o $@ $^

time-polynomial-detrender: time-polynomial-detrender.o misc.o unit_testing.o polynomial_detrender.o
	$(CPP) $(CPP_LFLAGS) -o $@ $^

time-upsample: time-upsample.o unit_testing.o upsample.o
	$(CPP) $(CPP_LFLAGS) -o $@ $^
