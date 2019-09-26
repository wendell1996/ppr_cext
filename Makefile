main: main.o libppr.a
	g++ -std=c++11 main.o libppr.a -o main
main.o: main.cpp
	g++ -std=c++11 -c main.cpp
libppr.a: ppr.o
	ar crs libppr.a ppr.o
ppr.o: ppr.cpp
	g++ -std=c++11 -c ppr.cpp
.PHONY: clean debug dlib
clean:
	rm -f *.so *.o main
debug:
	g++ -std=c++11 -g -c ppr.cpp
	ar crs libppr.a ppr.o
	g++ -std=c++11 -g -c main.cpp
	g++ -std=c++11 main.o libppr.a -o main
dlib:
	g++ -std=c++11 -fPIC -c ppr.cpp
	g++ -shared -o libppr.so ppr.o
