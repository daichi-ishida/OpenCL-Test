CXX					:= g++
BUILD_TYPE			:= Release

TARGET				:= main

SRCDIR				:= src
BUILDDIR			:= obj
TARGETDIR			:= bin
OUTPUTDIR			:= output

# extensions
SRCEXT				:= cpp
DEPEXT				:= d
OBJEXT				:= o

# flags
CXXFLAGS			:= -MMD -MP -std=c++11
CXX_DEBUG_FLAGS		:= -Wall -g -O0
CXX_RELEASE_FLAGS	:= -s -O2
LDFLAGS				:=
INCLUDE				:= -I/usr/local/cuda-9.1/include
LIBS				:= -L/usr/local/cuda-9.1/lib64 -lOpenCL


#---------------------------------------------------------------------------------
# DO NOT EDIT BELOW THIS LINE
#---------------------------------------------------------------------------------

ifeq ($(BUILD_TYPE),Release)
	CXXFLAGS += $(CXX_RELEASE_FLAGS)
else ifeq ($(BUILD_TYPE),Debug)
	CXXFLAGS += $(CXX_DEBUG_FLAGS)
else
	$(error buildtype must be release, debug, profile or coverage)
endif

sources			:= $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
objects			:= $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(subst $(SRCEXT),$(OBJEXT),$(sources)))
dependencies 	:= $(subst .$(OBJEXT),.$(DEPEXT),$(objects))

# Default Make
all: directories $(TARGETDIR)/$(TARGET)

run: all
	$(TARGETDIR)/$(TARGET)

# Remake
remake: cleaner all

# make directory
directories:
	@mkdir -p $(TARGETDIR)
	@mkdir -p $(BUILDDIR)
	@mkdir -p $(OUTPUTDIR)

# remove directory for intermediate products
clean:
	@$(RM) -rf $(BUILDDIR)

# remove directories for both intermediate and final products
cleaner: clean
	@$(RM) -rf $(TARGETDIR)

-include $(dependencies)

# generate binary by linking objects
$(TARGETDIR)/$(TARGET): $(objects)
	$(CXX) $(CXXFLAGS) -o $(TARGETDIR)/$(TARGET) $^ $(LIBS)

# generate objects by compiling sources
# save dependencies of source as .d
$(BUILDDIR)/%.$(OBJEXT): $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ -c $<

# Non-File Targets
.PHONY: all run remake clean cleaner