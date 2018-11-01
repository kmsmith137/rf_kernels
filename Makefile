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
  intensity_clipper.hpp \
  intensity_clipper_internals.hpp \
  mask_counter.hpp \
  mean_rms.hpp \
  mean_rms_internals.hpp \
  online_mask_filler.hpp \
  polynomial_detrender.hpp \
  polynomial_detrender_internals.hpp \
  quantize.hpp \
  spline_detrender.hpp \
  spline_detrender_internals.hpp \
  std_dev_clipper.hpp \
  std_dev_clipper_internals.hpp \
  unit_testing.hpp \
  upsample.hpp \
  upsample_internals.hpp \
  xorshift_plus.hpp 

OFILES = \
  downsample.o \
  intensity_clipper.o \
  mask_counter.o \
  mean_rms.o \
  misc.o \
  online_mask_filler.o \
  polynomial_detrender.o \
  quantize.o \
  spline_detrender.o \
  std_dev_clipper.o \
  upsample.o

TESTBINFILES = \
  test-downsample \
  test-upsample \
  test-intensity-clipper \
  test-mask-counter \
  test-std-dev-clipper \
  test-online-mask-filler \
  test-polynomial-detrender \
  test-quantize \
  test-spline-detrender

TIMEBINFILES = \
  time-downsample \
  time-intensity-clipper \
  time-memory-access-patterns \
  time-online-mask-filler \
  time-polynomial-detrender \
  time-quantize \
  time-spline-detrender \
  time-std-dev-clipper \
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
	if [ -d $(INCDIR)/rf_kernels ]; then rmdir $(INCDIR)/rf_kernels; fi

clean:
	rm -f $(TESTBINFILES) *~ *.o *.so *.pyc rf_kernels/*~ unittest_touchfiles/ut_*
	if [ -d unittest_touchfiles ]; then rmdir unittest_touchfiles; fi

unittest_touchfiles/ut_%: %
	mkdir -p unittest_touchfiles && ./$< && touch $@

librf_kernels.so: $(OFILES)
	$(CPP) $(CPP_LFLAGS) -shared -o $@ $^


####################################################################################################


CORE_DEPS = rf_kernels/core.hpp rf_kernels/internals.hpp

TEST_DEPS = $(CORE_DEPS) rf_kernels/unit_testing.hpp

DS_DEPS = rf_kernels/downsample.hpp rf_kernels/downsample_internals.hpp
US_DEPS = rf_kernels/upsample.hpp rf_kernels/upsample_internals.hpp
MR_DEPS = rf_kernels/mean_rms.hpp rf_kernels/mean_rms_internals.hpp $(DS_DEPS)
IC_DEPS = rf_kernels/intensity_clipper.hpp rf_kernels/intensity_clipper_internals.hpp $(MR_DEPS) $(US_DEPS)
SDC_DEPS = rf_kernels/std_dev_clipper.hpp rf_kernels/std_dev_clipper_internals.hpp $(MR_DEPS)


downsample.o: downsample.cpp $(CORE_DEPS) $(DS_DEPS)
	$(CPP) -c -o $@ $<

intensity_clipper.o: intensity_clipper.cpp $(CORE_DEPS) $(IC_DEPS)
	$(CPP) -c -o $@ $<

mask_counter.o: mask_counter.cpp $(CORE_DEPS) rf_kernels/mask_counter.hpp
	$(CPP) -c -o $@ $<

mean_rms.o: mean_rms.cpp $(CORE_DEPS) $(MR_DEPS)
	$(CPP) -c -o $@ $<

misc.o: misc.cpp $(CORE_DEPS)
	$(CPP) -c -o $@ $<

online_mask_filler.o: online_mask_filler.cpp $(CORE_DEPS) rf_kernels/xorshift_plus.hpp rf_kernels/online_mask_filler.hpp
	$(CPP) -c -o $@ $<

polynomial_detrender.o: polynomial_detrender.cpp $(CORE_DEPS) rf_kernels/polynomial_detrender.hpp rf_kernels/polynomial_detrender_internals.hpp
	$(CPP) -c -o $@ $<

quantize.o: quantize.cpp rf_kernels/quantize.hpp rf_kernels/internals.hpp
	$(CPP) -c -o $@ $<

spline_detrender.o: spline_detrender.cpp $(CORE_DEPS) rf_kernels/spline_detrender.hpp rf_kernels/spline_detrender_internals.hpp
	$(CPP) -c -o $@ $<

std_dev_clipper.o: std_dev_clipper.cpp $(CORE_DEPS) $(SDC_DEPS)
	$(CPP) -c -o $@ $<

unit_testing.o: unit_testing.cpp $(TEST_DEPS)
	$(CPP) -c -o $@ $<

upsample.o: upsample.cpp $(CORE_DEPS) $(US_DEPS)
	$(CPP) -c -o $@ $<


####################################################################################################


test-downsample.o: test-downsample.cpp $(TEST_DEPS) rf_kernels/downsample.hpp
	$(CPP) -c -o $@ $<

test-intensity-clipper.o: test-intensity-clipper.cpp $(TEST_DEPS) rf_kernels/upsample.hpp rf_kernels/downsample.hpp rf_kernels/mean_rms.hpp rf_kernels/intensity_clipper.hpp
	$(CPP) -c -o $@ $<

test-mask-counter.o: test-mask-counter.cpp $(TEST_DEPS) rf_kernels/mask_counter.hpp rf_kernels/internals.hpp
	$(CPP) -c -o $@ $<

test-online-mask-filler.o: test-online-mask-filler.cpp $(TEST_DEPS) rf_kernels/xorshift_plus.hpp rf_kernels/online_mask_filler.hpp
	$(CPP) -c -o $@ $<

test-polynomial-detrender.o: test-polynomial-detrender.cpp $(TEST_DEPS) rf_kernels/polynomial_detrender.hpp rf_kernels/polynomial_detrender_internals.hpp
	$(CPP) -c -o $@ $<

test-quantize.o: test-quantize.cpp $(TEST_DEPS) rf_kernels/quantize.hpp
	$(CPP) -c -o $@ $<

test-spline-detrender.o: test-spline-detrender.cpp $(TEST_DEPS) rf_kernels/spline_detrender.hpp rf_kernels/spline_detrender_internals.hpp
	$(CPP) -c -o $@ $<

test-std-dev-clipper.o: test-std-dev-clipper.cpp $(TEST_DEPS) rf_kernels/mean_rms.hpp rf_kernels/std_dev_clipper.hpp
	$(CPP) -c -o $@ $<

test-upsample.o: test-upsample.cpp $(TEST_DEPS) rf_kernels/upsample.hpp
	$(CPP) -c -o $@ $<


test-downsample: test-downsample.o downsample.o
	$(CPP) $(CPP_LFLAGS) -o $@ $^

test-intensity-clipper: test-intensity-clipper.o intensity_clipper.o upsample.o downsample.o mean_rms.o misc.o
	$(CPP) $(CPP_LFLAGS) -o $@ $^

test-mask-counter: test-mask-counter.o mask_counter.o
	$(CPP) $(CPP_LFLAGS) -o $@ $^

test-online-mask-filler: test-online-mask-filler.o online_mask_filler.o
	$(CPP) $(CPP_LFLAGS) -o $@ $^

test-polynomial-detrender: test-polynomial-detrender.o polynomial_detrender.o misc.o
	$(CPP) $(CPP_LFLAGS) -o $@ $^

test-quantize: test-quantize.o quantize.o
	$(CPP) $(CPP_LFLAGS) -o $@ $^

test-spline-detrender: test-spline-detrender.o spline_detrender.o
	$(CPP) $(CPP_LFLAGS) -o $@ $^

test-std-dev-clipper: test-std-dev-clipper.o std_dev_clipper.o misc.o mean_rms.o
	$(CPP) $(CPP_LFLAGS) -o $@ $^

test-upsample: test-upsample.o upsample.o
	$(CPP) $(CPP_LFLAGS) -o $@ $^


####################################################################################################


time-downsample.o: time-downsample.cpp $(TEST_DEPS) rf_kernels/downsample.hpp 
	$(CPP) -c -o $@ $<

time-intensity-clipper.o: time-intensity-clipper.cpp $(TEST_DEPS) rf_kernels/intensity_clipper.hpp
	$(CPP) -c -o $@ $<

time-memory-access-patterns.o: time-memory-access-patterns.cpp $(TEST_DEPS)
	$(CPP) -c -o $@ $<

time-online-mask-filler.o: time-online-mask-filler.cpp $(TEST_DEPS) rf_kernels/xorshift_plus.hpp rf_kernels/online_mask_filler.hpp
	$(CPP) -c -o $@ $<

time-polynomial-detrender.o: time-polynomial-detrender.cpp $(TEST_DEPS) rf_kernels/polynomial_detrender.hpp
	$(CPP) -c -o $@ $<

time-quantize.o: time-quantize.cpp $(TEST_DEPS) rf_kernels/quantize.hpp 
	$(CPP) -c -o $@ $<

time-spline-detrender.o: time-spline-detrender.cpp $(TEST_DEPS) rf_kernels/spline_detrender.hpp
	$(CPP) -c -o $@ $<

time-std-dev-clipper.o: time-std-dev-clipper.cpp $(TEST_DEPS) rf_kernels/std_dev_clipper.hpp
	$(CPP) -c -o $@ $<

time-upsample.o: time-upsample.cpp $(TEST_DEPS) rf_kernels/upsample.hpp
	$(CPP) -c -o $@ $<


time-downsample: time-downsample.o unit_testing.o downsample.o
	$(CPP) $(CPP_LFLAGS) -o $@ $^

time-intensity-clipper: time-intensity-clipper.o intensity_clipper.o unit_testing.o misc.o
	$(CPP) $(CPP_LFLAGS) -o $@ $^

time-memory-access-patterns: time-memory-access-patterns.o unit_testing.o
	$(CPP) $(CPP_LFLAGS) -o $@ $^

time-online-mask-filler: time-online-mask-filler.o online_mask_filler.o unit_testing.o
	$(CPP) $(CPP_LFLAGS) -o $@ $^

time-quantize: time-quantize.o quantize.o unit_testing.o
	$(CPP) $(CPP_LFLAGS) -o $@ $^

time-spline-detrender: time-spline-detrender.o unit_testing.o spline_detrender.o
	$(CPP) $(CPP_LFLAGS) -o $@ $^

time-std-dev-clipper: time-std-dev-clipper.o unit_testing.o std_dev_clipper.o misc.o
	$(CPP) $(CPP_LFLAGS) -o $@ $^

time-polynomial-detrender: time-polynomial-detrender.o misc.o unit_testing.o polynomial_detrender.o
	$(CPP) $(CPP_LFLAGS) -o $@ $^

time-upsample: time-upsample.o unit_testing.o upsample.o
	$(CPP) $(CPP_LFLAGS) -o $@ $^
