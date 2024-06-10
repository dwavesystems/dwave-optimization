ROOT := ../..
INCLUDE := $(ROOT)/dwave/optimization/include
SRC := $(ROOT)/dwave/optimization/src
CATCH2 := $(ROOT)/extern/catch2/single_include/
FLAGS := $(FLAGS) -std=c++20 -Wall -Wno-unknown-pragmas -O3
DEPDIR := .deps
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.d

OBJECTS := $(patsubst $(SRC)/%.cpp,%.o,$(wildcard $(SRC)/*.cpp))
NODES := $(patsubst $(SRC)/nodes/%.cpp,%.o,$(wildcard $(SRC)/nodes/*.cpp))
TESTS := $(patsubst tests/%.cpp,%.o,$(wildcard tests/*.cpp))

all: update tests

test_main.o:
	$(CXX) -std=c++11 test_main.cpp -c -I$(CATCH2)

$(DEPDIR): ; @mkdir -p $@

# Rules for the objects
$(OBJECTS): %.o: $(SRC)/%.cpp $(INCLUDE)/dwave-optimization/%.hpp $(DEPDIR)/%.d | $(DEPDIR)
	$(CXX) $(DEPFLAGS) $(FLAGS) -c $< -o $@ -I$(INCLUDE)

$(NODES): %.o: $(SRC)/nodes/%.cpp $(INCLUDE)/dwave-optimization/nodes/%.hpp $(DEPDIR)/%.d | $(DEPDIR)
	$(CXX) $(DEPFLAGS) $(FLAGS) -c $< -o $@ -I$(INCLUDE)

$(TESTS): %.o: tests/%.cpp $(DEPDIR)/%.d | $(DEPDIR)
	$(CXX) $(DEPFLAGS) $(FLAGS) -c $< -o $@ -I$(CATCH2) -I$(INCLUDE)

# The big one
tests.out: $(OBJECTS) $(NODES) $(TESTS) test_main.o
	$(CXX) $(FLAGS) $(OBJECTS) $(NODES) $(TESTS) test_main.o -o tests.out

tests: tests.out
	./tests.out

update:
	git submodule init
	git submodule update

clean:
	rm -f *.o *.out
	rm -rf $(DEPDIR)

DEPFILES := $(patsubst $(SRC)/%.cpp,$(DEPDIR)/%.d,$(wildcard $(SRC)/*.cpp)) $(patsubst tests/%.cpp,$(DEPDIR)/%.d,$(wildcard tests/*.cpp)) $(patsubst $(SRC)/nodes/%.cpp,$(DEPDIR)/%.d,$(wildcard $(SRC)/nodes/*.cpp))
$(DEPFILES):
include $(wildcard $(DEPFILES))
