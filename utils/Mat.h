#ifndef VERTICALMPL_MAT_H
#define VERTICALMPL_MAT_H
#include <Eigen/Dense>
#include <Eigen/Core>
#include "Constant.h"
#include <random>
using namespace std;
using namespace Eigen;

typedef Matrix<u64, Dynamic, Dynamic> MatrixXu;
typedef Matrix<bool, Dynamic, Dynamic> MatrixXb;
typedef Matrix<u64, Dynamic, 1> MatrixXu1;
typedef Matrix<int, Dynamic, Dynamic> MatrixXi;

class Mat
{
public:
    static MatrixXu randomMatrixXu(int r, int c);
    static MatrixXu initFromVector(vector<u64> &data, int rows, int cols);
    static void truncateMatrixXu(MatrixXu &x);
    static MatrixXu truncateMat(const MatrixXu &x);
    static void reshape(MatrixXu &x, long nrows, long ncols);
    static void randomFill(MatrixXu &x);
    static MatrixXu constant_multiply(MatrixXu &x, double d);
    static MatrixXu constant_multiply(MatrixXu &x, u64 d);
    static MatrixXu cwiseProduct(MatrixXu &a, MatrixXu &b);
    static void residual(MatrixXi &a);
    static MatrixXu getFrom_pos(char *&p);
    static MatrixXd getFrom_pos_double(char *&p);
    static int toString_pos(char *p, int r, int c, vector<u64> val);
    static int toString_pos(char *p, int r, int c, vector<double> val);
    static vector<u64> toVector(MatrixXu &a);
    static MatrixXu fromVector(vector<u64> val, int r, int c);
    static void Relu(MatrixXu &x);
    static MatrixXu DRelu(MatrixXu &x);
    static void Sigmoid(MatrixXu &x);
    static MatrixXu DSigmoid(MatrixXu &x);
    static MatrixXb MatrixXb_xor(const MatrixXb &a, const MatrixXb &b);
};
#endif // VERTICAL_MAT_H
