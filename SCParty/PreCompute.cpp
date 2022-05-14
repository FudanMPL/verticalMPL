#include "PreCompute.h"
void PreCompute::getRandomu64(u64 &r)
{
    r = 0;
}

void PreCompute::getRandomMatrix(MatrixXu &a)
{
    a.setZero();
    // Mat::randomFill(a);
}

void PreCompute::getRandomnessforMultiply(MatrixXu &r, MatrixXu &r2, MatrixXu &s1, int pos)
{ 
    if (pos == 0)
    {
        r.setZero();
        r2.setZero();
        s1.setZero(); // random matrix related to the other party
    }
    else
    {
        r.setZero();
        r2.setZero();
        s1.setZero(); 
    }
}

void PreCompute::getRandomnessforShareMultiplyV(MatrixXu &r1, MatrixXu &r2, MatrixXu &rr2, MatrixXu &ss1, int pos)
{
    if (pos == 0)
    {
        r1.setZero();
        r2.setZero();
        rr2.setZero();
        ss1.setZero();
    }
    else
    {
        r1.setZero();
        r2.setZero();
        rr2.setZero();
        ss1.setZero();
    }
}

void PreCompute::getRandomnessforShareMultiplyM(MatrixXu &r1, MatrixXu &r2, MatrixXu &rr2, MatrixXu &rr1, MatrixXu &ss1, MatrixXu &ss2)
{
    r1.setZero();
    r2.setZero();
    rr2.setZero();
    rr1.setZero();
    ss1.setZero();
    ss2.setZero();
}
void PreCompute::getRandomnessforSharecwiseProduct(vector<MatrixXi> &r1, MatrixXi &r2, vector<MatrixXi> &rr2, vector<MatrixXi> &ss1, int rows, int cols, int pos)
{
    int size = r1.size();
    if (pos == 0)
    {
        r2.setZero();
        for (int i = 0; i < size; i++)
        {
            r1[i].resize(rows, cols);
            rr2[i].resize(rows, cols);
            ss1[i].resize(rows, cols);
            r1[i].setZero();
            rr2[i].setZero();
            ss1[i].setZero();
        }
    }
    else
    {
        r2.setZero();
        for (int i = 0; i < size; i++)
        {
            r1[i].resize(rows, cols);
            rr2[i].resize(rows, cols);
            ss1[i].resize(rows, cols);
            r1[i].setZero();
            rr2[i].setZero();
            ss1[i].setZero();
        }
    }
}

void PreCompute::getRandomnessforSharecwiseProduct(MatrixXu &r1, MatrixXu &r2, MatrixXu &ss1, int pos)
{
    if (pos == 0)
    {
        r1.setZero();
        r2.setZero();
        ss1.setZero();
    }
    else
    {
        r1.setZero();
        r2.setZero();
        ss1.setZero();
    }
}

void PreCompute::getRandomnessforSharecwiseProduct(MatrixXi &r1, MatrixXi &r2, MatrixXi &ss1, int pos)
{
    //randomess for sharecwiseproduct in small field
    if (pos == 0)
    {
        r1.setZero();
        r2.setZero();
        ss1.setZero();
    }
    else
    {
        r1.setZero();
        r2.setZero();
        ss1.setZero();
    }
}

void PreCompute::getRandomnessforWrap(MatrixXi &xi, MatrixXu1 &x, vector<int> &w)
{
    xi.setZero();
    x.setZero();
    fill(w.begin(), w.end(), 0);
}

void PreCompute::getRandomBitShare(MatrixXb &b, MatrixXi &b_share)
{
    b.setZero();
    b_share.setZero();
}

void PreCompute::getRandomBitShare(MatrixXb &b, MatrixXu &b_share)
{
    b.setZero();
    b_share.setZero();
}