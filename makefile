CC = nvcc
CFILES = $(shell find . -name '*.cu' -o -name '*.cpp')
OBJECTS = $(patsubst %.cpp, %.o, $(CFILES))
OBJECTS := $(patsubst %.cu, %.o, $(OBJECTS))

$(info $(OBJECTS))

CFLAGS= -lSDL2 -lSDL2main  --compiler-options -Wall

raytracer: $(OBJECTS)
	$(CC)  $(LDFLAGS) $(LIBS) -o raytracer $(OBJECTS) $(CFLAGS)

MAKE_OBJECT= $(CC) $(CFLAGS) -c $< -o $@
%.o: %.cpp
	$(MAKE_OBJECT)

%.o: %.cu
	$(MAKE_OBJECT)


run: raytracer
	./raytracer

profile: raytracer
	nvprof ./raytracer

clean :
	rm src/*.o
	rm raytracer