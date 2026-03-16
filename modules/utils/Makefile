FCOMP   = ifort
#gfortran
F2PY    = f2py

FCFLAGS =

INCLUDES =
LIBDIRS  = -L/star/lib
LIBS     = -lsla

all: pysla

pysla:
	$(F2PY) -c \
		--f77exec=$(FCOMP) \
		--f90exec=$(FCOMP) \
		$(FCFLAGS) \
		-m pysla pysla.f90 \
		$(INCLUDES) $(LIBDIRS) $(LIBS) 
