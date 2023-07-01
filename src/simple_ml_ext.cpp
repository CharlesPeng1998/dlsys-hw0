#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <cstring>
#include <iostream>

namespace py = pybind11;

// nk * km -> nm
float* matrix_mutiply(const float *x0, const float *x1, size_t n, size_t k, size_t m) {
    float* res = new float[n * m]();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            float sum = 0;
            for (int kk = 0; kk < k; ++kk) {
                sum += x0[i * k + kk] * x1[kk * m + j];
            }
            res[i * m + j] = sum;
        }
    }
    return res;
}

void exp(float* x, size_t n, size_t m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            float v = x[i * m + j];
            x[i * m + j] = std::exp(v);
        }
    }
}

void normalize(float* x, size_t n, size_t m) {
    for (int i = 0; i < n; ++i) {
        float sum = 0;
        for (int j = 0; j < m; ++j) {
            sum += x[i * m + j];
        }
        for (int j = 0; j < m; ++j) {
            x[i * m + j] /= sum;
        }
    }
}

float* one_hot(const unsigned char *y, size_t m, size_t k) {
    float* res = new float[m * k]();
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            if (j == y[i]) res[i * k + j] = 1.0;
            else res[i * k + j] = 0;
        }
    }
    return res;
}

float* transpose(const float *x, size_t n, size_t m) {
    float* res = new float[n * m]();
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            res[i * n + j] = x[j * m + i];
        }
    }
    return res;
}

void subtract(float *x, float *y, size_t n, size_t m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            x[i * m + j] -= y[i * m + j];
        }
    }
}

void div_scalar(float *x, float y, size_t n, size_t m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            x[i * m + j] /= y;
        }
    }
}

void mul_scalar(float *x, float y, size_t n, size_t m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            x[i * m + j] *= y;
        }
    }
}


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */
    for (int i = 0; i <= m - batch; i += batch) {
        const float* x_batch = X[i * n];
        const unsigned char* y_batch = y[i];
        float* Z = matrix_mutiply(x_batch, theta, batch, n, k); // mn * nk = mk
        exp(Z, batch, k);
        normalize(Z, batch, k);
        float* I_y = one_hot(y_batch, batch, k);
        subtract(Z, I_y, batch, k);
        float* gradient = matrix_mutiply(transpose(x_batch, batch, n), Z, n, batch, k);
        div_scalar(gradient, batch, n, k);
        mul_scalar(gradient, lr, n, k);
        subtract(theta, gradient, n, k);
    }
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
