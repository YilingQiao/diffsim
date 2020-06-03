CC := gcc
CXX := g++

# # uncomment to disable OpenGL functionality
# NO_OPENGL := true

CXXFLAGS := -Idependencies/include -I/opt/local/include
ifdef NO_OPENGL
	CXXFLAGS := $(CXXFLAGS) -DNO_OPENGL
endif
CXXFLAGS_DEBUG := -Wall -g -Wno-sign-compare
CXXFLAGS_RELEASE := -O3 -Wreturn-type -fopenmp
LDFLAGS := -Ldependencies/lib -L/opt/local/lib  -lz -ltaucs -llapack -lblas -lboost_filesystem-mt -lboost_system-mt -lboost_thread-mt -ljson -lgomp -lalglib
ifndef NO_OPENGL
	LDFLAGS := $(LDFLAGS) -lglut -lGLU -lGL
endif

OBJ := \
	auglag.o \
	bah.o \
	bvh.o \
	cloth.o \
	collision.o \
	collisionutil.o \
	conf.o \
	constraint.o \
	dde.o \
	display.o \
	displayphysics.o \
	displayreplay.o \
	displaytesting.o \
	dynamicremesh.o \
	geometry.o \
	handle.o \
	io.o \
	lbfgs.o \
	lsnewton.o \
	magic.o \
	main.o \
	mesh.o \
	misc.o \
	morph.o \
	mot_parser.o \
	nearobs.o \
	nlcg.o \
	obstacle.o \
	physics.o \
	popfilter.o \
	plasticity.o \
	proximity.o \
	remesh.o \
	runphysics.o \
	separate.o \
	separateobs.o \
	simulation.o \
	spline.o \
	strainlimiting.o \
	taucs.o \
	tensormax.o \
	timer.o \
	transformation.o \
	trustregion.o \
	util.o \
	vectors.o

.PHONY: all debug release tags clean

release:

all: debug release ctags

debug: bin/arcsimd ctags

release: bin/arcsim ctags

bin/arcsimd: $(addprefix build/debug/,$(OBJ))
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)

bin/arcsim: $(addprefix build/release/,$(OBJ))
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)

build/debug/%.o: src/%.cpp
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $(CXXFLAGS_DEBUG) $< -o $@

build/release/%.o: src/%.cpp
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $(CXXFLAGS_RELEASE) $< -o $@

# Nicked from http://www.gnu.org/software/make/manual/make.html#Automatic-Prerequisites
build/dep/%.d: src/%.cpp
	@set -e; rm -f $@; \
	$(CXX) -MM $(CXXFLAGS) $< -o $@.tmp; \
	sed 's,\($*\)\.o[ :]*,build/debug/\1.o build/release/\1.o: ,g' < $@.tmp > $@; \
	sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
	    -e '/^$$/ d' -e 's/$$/ :/' < $@.tmp >> $@; \
	rm -f $@.tmp

-include $(addprefix build/dep/,$(OBJ:.o=.d))

ctags:
	cd src; ctags -w *.?pp
	cd src; etags *.?pp

clean:
	rm -rf bin/* build/debug/* build/release/*
