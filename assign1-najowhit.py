import numpy
import sys


if len(sys.argv) != 2:
    print("Supply a filename as an argument")
    exit(1)

dataFile = sys.argv[1]

data = numpy.loadtxt(dataFile, delimiter=",", usecols=(0, 1, 2, 4, 5, 6, 7, 8, 9))
print(data)

