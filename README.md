### RF_KERNELS

Fast C++/assembly kernels for RFI removal and related tasks.
This is a core library for
[CHIMEFRB/bonsai](https://github.com/CHIMEFRB/bonsai) and
[kmsmith137/rf_pipelines](https://github.com/kmsmith137/rf_pipelines).

Some day, rf_kernels will be systematically documented!
For now, here are its installation instructions.

### INSTALLATION

  - The rf_kernels Makefile assumes the existence of a file `Makefile.local` which defines
    the following machine-dependent Makefile variables:
    ```
      INCDIR     Installation directory for C++ header files
      LIBDIR     Installation directory for libraries
      CPP        C++ compiler executable + flags, see below for tips!
    ```

    Rather than write a Makefile.local from scratch, I recommend that you start with one of the
    examples in the site/ directory, which contains Makefile.locals for a few frequently-used
    CHIME machines.  In particular, site/Makefile.local.kms_laptop16 is a recent osx machine,
    and site/Makefile.local.frb1 is a recent CentOS Linux machine.  (If you're a member of
    CHIME and you're using one of these machines, you can just symlink the appropriate file in
    site/ to ./Makefile.local)

  - Do `make all install` to build.

  - Do `make test` if you want to run some unit tests.
  
  - If you have trouble getting rf_kernels to build/work, then the problem probably has
    something to do with your compiler flags (specified as part of CPP) or environment 
    variables.  Here are a few hints:

      - You probably need `-std=c++11` in your compiler flags, for C++11 support
      - You probably need `-pthread` in your compiler flags, in order to compile
        some multithreaded timing tests.
      - You probably need `-march=native` in your compiler flags, to get AVX/AVX2
        intrinsics.  (I usually use optimization flags `-O3 -march=native -ffast-math -funroll-loops`.)
      - If you're compiling with gcc, I recommend adding '--param inline-unit-growth=10000' to the command line.
        This allows more aggressive inlining, and improves performance significantly, at least on the CHIME compute nodes.
      - You probably want `-Wall -fPIC` in your compiler flags on general principle.
      - The rf_kernels build procedure assumes that the current directory is searched for header
        files and libraries, i.e. you should have `-I. -L.` in your compiler flags.
      - You also probably want `-I$(INCDIR) -L$(LIBDIR)` in your compiler flags, so that
        these install dirs are also searched for headers/libraries (e.g. simpulse)
      - You may need more -I and -L flags to find all necessary headers/libraries.
      - If everything compiles but libraries are not being found at runtime, then you
        probably need to add `.` or LIBDIR to the appropriate environment variable
        ($LD_LIBRARY_PATH in Linux, or $DYLD_LIBRARY_PATH in osx)

    Feel free to email me if you have trouble!
