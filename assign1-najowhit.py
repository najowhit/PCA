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


# Helper method for centering data
def centerData(data):
    row, columns = numpy.shape(data)
    one_vector = numpy.ones((row, columns))

    mean = data.mean(axis=0)

    z_center = data - one_vector * mean

    return z_center


def computeCov(data):

    # Center the data, this redundancy from the centerData() method is kept to keep the algorithm clear
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

    return my_cov


def varianceProjection(data):

    # Eigenvector that corresponds to largest eigenvalue = largest eigenvector

    numpy_cov = numpy.cov(data, rowvar=0, bias=1)
    val, vec = linalg.eig(numpy_cov)

    # Eigenvectors are the columns of vec, transpose to align eigenvalues with eigenvectors
    # Before a transpose vec[0] will return a row instead of the needed column
    t_vec = vec.transpose()

    # Find largest eigenvalue
    sorted_val = numpy.argsort(val)

    # Get the index that corresponds to largest eigenvalues
    first = sorted_val[len(sorted_val) - 1]
    second = sorted_val[len(sorted_val) - 2]

    # Get the largest two eigenvectors using the indicies
    largest = t_vec[first]
    snd_largest = t_vec[second]

    # Compute the projection
    subspace = numpy.array([largest, snd_largest])
    z_center = centerData(data)
    projection = numpy.dot(z_center, subspace.transpose())

    # Compute variance of projection
    my_cov_projection = computeCov(projection)
    variance = numpy.trace(my_cov_projection)
    print(variance)


def lambdaEig(data):

    # Eigenvector that corresponds to largest eigenvalue = largest eigenvector

    numpy_cov = numpy.cov(data, rowvar=0, bias=1)
    val, vec = linalg.eig(numpy_cov)

    # Decomposition form = UlambdaU^T ,  U = vec in this case
    # Read 7.2.4 to figure plan from here
    val_lambda = numpy.diag(val)
    sigma = numpy.dot(numpy.dot(vec, val_lambda), vec.transpose())

    print(sigma)


def pca(data, alpha):

    # 1) Compute mean
    mean = data.mean(axis=0)

    # 2) Center the data
    row, columns = numpy.shape(data)
    one_vector = numpy.ones((row, columns))
    z_center = data - one_vector * mean

    # 3) Compute covariance matrix
    numpy_cov = numpy.cov(z_center, rowvar=0, bias=1)

    # 4) Compute eigenvalues & # 5) Compute eigenvectors
    val, vec = linalg.eig(numpy_cov)


    # 6) Fraction of total variance

    # 7) Choose dimensionality

    # 8) Reduced basis

    # 9) Reduced dimensionality data


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
    varianceProjection(original_data)
    lambdaEig(original_data)



main()


