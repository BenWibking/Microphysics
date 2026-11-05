PRECISION  = DOUBLE
PROFILE    = FALSE

DEBUG      = FALSE

DIM        = 3

COMP	   = gnu

USE_MPI    = FALSE
USE_OMP    = FALSE

USE_REACT = TRUE

EBASE = main

BL_NO_FORT=TRUE

# define the location of the Microphysics top directory
MICROPHYSICS_HOME  := ../..

# This sets the EOS directory
EOS_DIR     := helmholtz

# This sets the network directory
NETWORK_DIR := aprox13

CONDUCTIVITY_DIR := stellar

EXTERN_SEARCH += .

Bpack   := ./Make.package
Blocs   := .

# Loom burn kernel (loom-burn-kernel skill). LOOM_KERNEL_DIR is the port's
# kernel directory relative to test_react, e.g. loom/<network>_<integrator>.
ifneq ($(filter TRUE,$(USE_LOOM_VALIDATE) $(USE_LOOM_BURN)),)
  ifeq ($(LOOM_KERNEL_DIR),)
    $(error USE_LOOM_VALIDATE/USE_LOOM_BURN need LOOM_KERNEL_DIR=loom/<network>_<integrator>)
  endif
  # Unquoted on purpose: the build scripts get $(DEFINES) inside "...". The
  # harness headers stringize it (LOOM_KERNEL_PATH in loom_eos_params.H), so
  # the path must not contain a predefined macro name such as linux or unix.
  ifneq ($(findstring linux,$(LOOM_KERNEL_DIR))$(findstring unix,$(LOOM_KERNEL_DIR)),)
    $(error LOOM_KERNEL_DIR must not contain "linux" or "unix" (predefined macros))
  endif
  DEFINES += -DLOOM_KERNEL_DIR_PATH=$(LOOM_KERNEL_DIR)
endif

# native validation of the Loom kernel translation (loom_validate.cpp)
ifeq ($(USE_LOOM_VALIDATE), TRUE)
  DEFINES += -DLOOM_VALIDATE
  VPATH_LOCATIONS += $(LOOM_KERNEL_DIR)
endif

# HIP launcher for the Loom burn kernel (loom_launch.H; runtime parameter
# unit_test.loom_hsaco selects it, the stock burn stays the default)
ifeq ($(USE_LOOM_BURN), TRUE)
  ifneq ($(USE_HIP), TRUE)
    $(error USE_LOOM_BURN=TRUE requires USE_HIP=TRUE)
  endif
  DEFINES += -DMICROPHYSICS_LOOM
endif

include $(MICROPHYSICS_HOME)/unit_test/Make.unit_test

# The kernel directory goes LAST on the include path: its eos.H, burner.H,
# ... share names with stock Microphysics headers, which must keep winning
# `#include <eos.H>`. Kernel headers include each other with "..." (same
# directory first); validators are reached as <validate_<module>.H>.
ifeq ($(USE_LOOM_VALIDATE), TRUE)
  INCLUDE_LOCATIONS += $(LOOM_KERNEL_DIR)
endif

