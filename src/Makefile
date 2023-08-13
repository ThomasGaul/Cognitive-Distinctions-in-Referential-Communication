main: main.o CTRNN.o TSearch.o Agent.o random.o
	g++ -pthread -o main main.o CTRNN.o TSearch.o Agent.o random.o
main.o: main.cpp CTRNN.h Agent.h TSearch.h
	g++ -pthread -c -O3 main.cpp
Agent.o: Agent.cpp Agent.h TSearch.h CTRNN.h random.h VectorMatrix.h
	g++ -pthread -c -O3 Agent.cpp
TSearch.o: TSearch.cpp TSearch.h
	g++ -pthread -c -O3 TSearch.cpp
CTRNN.o: CTRNN.cpp CTRNN.h random.h VectorMatrix.h
	g++ -pthread -c -O3 CTRNN.cpp
random.o: random.cpp random.h VectorMatrix.h
	g++ -pthread -c -O3 random.cpp	
clean:
	rm *.o main