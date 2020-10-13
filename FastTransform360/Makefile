all: unittest libft360.so

unittest: unittest.o libft360.so
	g++ -o a.out unittest.o libft360.so `pkg-config --libs opencv`

unittest.o: unittest.cc
	g++ -I/usr/local/cuda-9.0/include -c unittest.cc

libft360.so: transformer.o cuda_transformer.o
	g++ -shared -o libft360.so transformer.o cuda_transformer.o -lcudart

transformer.o: transformer.cc transformer.h
	g++ -I/usr/local/cuda-9.0/include -fPIC -c transformer.cc

cuda_transformer.o: cuda_transformer.cu
	nvcc --compiler-options -fPIC -c cuda_transformer.cu

clean:
	rm -f *.o *.out *.png
