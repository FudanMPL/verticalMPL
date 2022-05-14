#ifndef VERTICALMPL_PRECOMPUTE_H
#define VERTICALMPL_PRECOMPUTE_H

#include <vector>
#include "../utils/Mat.h"

extern int party;
class PreCompute
{
public:
    static void getRandomMatrix(MatrixXu &a);
    static void getRandomu64(u64 &r);
    static void getRandomnessforMultiply(MatrixXu &r, MatrixXu &r2, MatrixXu &s1, int pos);
    static void getRandomnessforShareMultiplyV(MatrixXu &r1, MatrixXu &r2, MatrixXu &rr2, MatrixXu &ss1, int pos);
    static void getRandomnessforShareMultiplyM(MatrixXu &r1, MatrixXu &r2, MatrixXu &rr2, MatrixXu &rr1, MatrixXu &ss1, MatrixXu &ss2);
    static void getRandomnessforSharecwiseProduct(vector<MatrixXi> &r1, MatrixXi &r2, vector<MatrixXi> &rr2, vector<MatrixXi> &ss1, int rows, int cols, int pos);
    static void getRandomnessforSharecwiseProduct(MatrixXu &r1, MatrixXu &r2, MatrixXu &ss1, int pos);
    static void getRandomnessforSharecwiseProduct(MatrixXi &r1, MatrixXi &r2, MatrixXi &ss1, int pos);
    static void getRandomnessforWrap(MatrixXi &xi, MatrixXu1 &x, vector<int> &w);
    static void getRandomBitShare(MatrixXb &b, MatrixXi &b_share);
    static void getRandomBitShare(MatrixXb &b, MatrixXu &b_share);
};
#endif