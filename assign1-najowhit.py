import numpy
import sys

'''
Example from slides replicated

matrix = numpy.array([[5, 3], [1, 4]])
#square = numpy.shape(matrix)

oneV = numpy.ones(width)
#print(oneV)

mean = matrix.mean(axis=0)
#print(mean)

center = matrix - oneV*mean
print(center)
'''

def computeCov(data):
    # Center the data
    square = numpy.shape(data)
    one_vector = numpy.ones(square)

    mean = data.mean(axis=0)

    z_center = data - one_vector*mean

    # Compute covariance

    # Compare covariance with numpy.cov'''

    numpy_cov = numpy.cov(z_center, rowvar=0, bias=1)



def main():

    if len(sys.argv) != 2:
        print("Supply a filename as an argument")
        exit(1)

    dataFile = sys.argv[1]

    try:
        originalData = numpy.loadtxt(dataFile, delimiter=",", usecols=(0, 1, 2, 4, 5, 6, 7, 8, 9))

    except IOError:
        print("Error loading file")
        exit(1)

    computeCov(originalData)


main()


