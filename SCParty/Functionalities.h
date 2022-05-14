#ifndef VERTICALMPL_FUNCTIONALITIES_H
#define VERTICALMPL_FUNCTIONALITIES_H

#include "PreCompute.h"
#include "../utils/SocketManager.h"

extern SocketManager::VMPL tel;

class Functionalities
{
public:
    static MatrixXb privateCompare(const MatrixXi &xi, const MatrixXu &r, const MatrixXb &b, const MatrixXi &b_share);
    static MatrixXu B2A(const MatrixXb &b);
    static MatrixXu selectshare(MatrixXu &a, MatrixXb &b);
    static MatrixXb wrap(MatrixXi &xi, MatrixXu1 &x, vector<int> alpha, const MatrixXu &a);
    static MatrixXb MSB(const MatrixXu &a);
    static MatrixXu DRelu(const MatrixXu &a);
    static MatrixXu Relu(MatrixXu &wx);
    static void Relu(MatrixXu &wx, MatrixXu &dr, MatrixXu &r);
    static void DSigmoid(MatrixXu &wx, MatrixXu &ds, MatrixXu &s);
    static MatrixXu Sigmoid(MatrixXu &wx);
    static MatrixXu Multiply(const MatrixXu &a, int rows, int cols, int pos);
    static MatrixXu ShareMultiply(const MatrixXu &x, const MatrixXu &y);
    static MatrixXu Multiply_Matrix(const MatrixXu &x, int rows, int cols, int pos);
    static MatrixXu ShareMultiply_Vector(const MatrixXu &x, const MatrixXu &y);
    static MatrixXu ShareMultiply_Matrix(const MatrixXu &x, const MatrixXu &y);
    static MatrixXu SharecwiseProduct(const MatrixXu &x, const MatrixXu &y);
    static MatrixXi SharecwiseProduct(const MatrixXi &x, const MatrixXi &y);
    static void SharecwiseProduct(const vector<MatrixXi> &v, const MatrixXi &w1, vector<MatrixXi> &u);
    static MatrixXu reveal(const MatrixXu &a, int target);
    static MatrixXu reveal(const MatrixXu &a);
    static MatrixXb reveal(const MatrixXb &b);
    static MatrixXi reveal(const MatrixXi &a);
};

#endif