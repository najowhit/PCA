import numpy
import sys
from numpy import linalg


''''# Center the data
data = numpy.array([[1,4], [3,8], [7, 2], [6,1], [3,0]])
length, square = numpy.shape(data)
one_vector = numpy.ones((length, square))

mean = data.mean(axis=0)

z_center = data - one_vector*mean


# Compute covariance

# Compare covariance with numpy.cov

numpy_cov = numpy.cov(z_center, rowvar=0, bias=1)
print(numpy_cov)

my_cov = 1/float(length) * numpy.dot(z_center.transpose(), z_center)
print(my_cov)'''

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
    row, columns = numpy.shape(data)
    one_vector = numpy.ones((row, columns))

    mean = data.mean(axis=0)

    z_center = data - one_vector*mean

    # Compute covariance
    # covariance =  zTz / n
    my_cov = numpy.dot(z_center.transpose(), z_center) / (row * 1.00)

    # Compare covariance with numpy.cov

    numpy_cov = numpy.cov(z_center, rowvar=0, bias=1)

    cov_equality = numpy.allclose(my_cov, numpy_cov)
    print('my_cov = numpy_cov')
    print(cov_equality)


def dEig(data):

    # Eigenvector that corresponds to largest eigenvalue = largest eigenvector

    numpy_cov = numpy.cov(data, rowvar=0, bias=1)
    val, vec = linalg.eig(numpy_cov)

    # Eigenvectors are the columns of vec
    t_vec = vec.transpose()

    # Find largest eigenvalue

    sorted_val = numpy.argsort(val)
    print(sorted_val)

    # Get the index that corresponds to largest eigenvalues
    first = sorted_val[len(sorted_val) - 1]
    second = sorted_val[len(sorted_val) - 2]
    print(t_vec[first])



def main():

    if len(sys.argv) != 2:
        print("Supply a filename as an argument")
        exit(1)

    data_file = sys.argv[1]

    try:
        original_data = numpy.loadtxt(data_file, delimiter=",", usecols=(0, 1, 2, 4, 5, 6, 7, 8, 9))

    except IOError:
        print("Error loading file")
        exit(1)

    computeCov(original_data)
    dEig(original_data)


main()


