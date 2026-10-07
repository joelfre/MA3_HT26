""" MA3.py

Student: Joel Frennesson
Mail:
Reviewed by:
Date reviewed:

"""
import random
import matplotlib.pyplot as plt
import math as m
import concurrent.futures as future
from statistics import mean 
from time import perf_counter as pc
import functools
from numba import njit

# Exc1
def approximate_pi(n):
    # n is the number of points
    circle = 0
    circle_list = [[], []]
    square = 0
    square_list = [[], []]

    for i in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if m.sqrt(x**2 + y**2) <= 1:
            circle += 1
            circle_list[0] += [x]
            circle_list[1] += [y]
        else:
            square += 1
            square_list[0] += [x]
            square_list[1] += [y]

    print(4*(circle/(circle + square)))
    plt.scatter(circle_list[0], circle_list[1], c = "red")
    plt.scatter(square_list[0], square_list[1], c = "blue")
    plt.axis("equal")
    plt.savefig(f"n={n}")

    return 4*(circle/(circle + square))

# Exc2, approximation
def sphere_volume(n, d): 
    # n is the number of points
    # d is the number of dimensions of the sphere 
    point_list = [[random.uniform(-1, 1) for x in range(d)] for i in range(n)] #Nested list comprehension för att skapa lista med alla koordinater
    sum_of_squares = lambda lst: functools.reduce(lambda x, y: x+y, list(map(lambda x: x**2, lst))) #Summan av x1^2 + x2^2...
    inside_list = [x for x in point_list if sum_of_squares(x) <= 1]

    return (2**d)*len(inside_list)/n #2^d är volymen på hyperkuben med sida 2r och r = 1

#Exc2, real value
def hypersphere_exact(n, d):
    # n is the number of points
    # d is the number of dimensions of the sphere 
    return (m.pi**(d/2))/(m.gamma(d/2 + 1))

#Exc3: numba version
@njit
def sphere_volume_numba(n:int, d:int)->float:
    # n is the number of points
    # d is the number of dimensions of the sphere
    sum = 0
    inside = 0
    point_list = [[random.uniform(-1, 1) for x in range(d)] for i in range(n)]
    for x in point_list:
        for i in x:
            sum += i**2
        if sum <= 1:
            inside += 1
        sum = 0
    #np is the number of processes
    return (2**d)*inside/n

#Exc4: parallel code - parallelize actual computations by splitting data
def sphere_volume_parallel(n, d, np=10):
    # n is the number of points
    # d is the number of dimensions of the sphere
    # np is the number of processes
    n_list = [n//np for i in range(np)]
    d_list = [d for i in range(np)]
    with future.ProcessPoolExecutor() as ex:
        result = list(ex.map(sphere_volume, n_list, d_list))
    return mean(result) #Mean from statistics
    
def main():
    # Exc1
    dots = [1000, 10000, 100000]
    for n in dots:
        approximate_pi(n)

    # Exc2
    n = 100000
    d = 2
    print(f"Monte carlo hypersphere = {sphere_volume(n, d)}")
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")

    n = 100000
    d = 11
    sphere_volume(n, d)
    print(f"Monte carlo hypersphere = {sphere_volume(n, d)}")
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")

    # Exc3
    n = 1000000
    d = 11
    start = pc()
    sphere_volume(n, d)
    stop = pc()
    print(f"Exc3: Sequential time of {d} and {n}: {stop-start}")
    print("What is numba time?")
    start = pc()
    sphere_volume_numba(n, d)
    stop = pc()
    print(f"Numba time of {d} and {n}: {stop-start}")
    

    # Exc4
    n = 1000000
    d = 11
    start = pc()
    sphere_volume(n, d)
    stop = pc()
    print(f"Exc4: Sequential time of {d} and {n}: {stop-start}")
    print("What is parallel time?")
    start = pc()
    print(sphere_volume_parallel(n, d, 4))
    stop = pc()
    print(f"Parallell time of {d} and {n}: {stop-start}")
    
    

if __name__ == '__main__':
	main()
