#include "Mat.h"

MatrixXu Mat::randomMatrixXu(int rows, int cols)
{
    MatrixXu res = MatrixXu::Zero(rows, cols);
    u64 *data = res.data();
    for (int i = 0; i < res.size(); i++)
    {
        *(data + i) = Constant::Util::random_u64();
    }
    return res;
}

MatrixXu Mat::initFromVector(vector<u64> &data, int rows, int cols)
{
    assert(data.size() == (ulong)(rows * cols));
    MatrixXu res = MatrixXu::Zero(rows, cols);
    uint64_t *pt = res.data();
    for (long i = 0; i < res.size(); i++)
    {
        *(pt + i) = data[i];
    }
    return res;
}

void Mat::truncateMatrixXu(MatrixXu &x)
{
    for (long i = 0; i < x.size(); i++)
    {
        x.data()[i] = Constant::Util::truncate(x.data()[i]);
    }
}

MatrixXu Mat::truncateMat(const MatrixXu &x)
{
    MatrixXu res(x.rows(), x.cols());

    for (long i = 0; i < x.size(); i++)
    {
        res.data()[i] = Constant::Util::truncate(x.data()[i]);
    }
    return res;
}

void Mat::reshape(MatrixXu &x, long nrows, long ncols)
{
    if (x.rows() == nrows && x.cols() == ncols)
    {
        return;
    }
    MatrixXu nx = MatrixXu::Zero(nrows, ncols);
    nx.topLeftCorner(x.rows(), x.cols()) = x;
    x = nx;
}

void Mat::randomFill(MatrixXu &x)
{
    for (long i = 0; i < x.size(); i++)
    {
        x.data()[i] = Constant::Util::random_u64();
    }
}

MatrixXu Mat::constant_multiply(MatrixXu &x, double d)
{
    MatrixXu res(x.rows(), x.cols());
    for (int i = 0; i < x.size(); i++)
    {
        res.data()[i] = (x.data()[i]) * Constant::Util::double_to_u64(d);
    }
    truncateMatrixXu(res);
    return res;
}

MatrixXu Mat::constant_multiply(MatrixXu &x, u64 d)
{
    MatrixXu res(x.rows(), x.cols());
    for (int i = 0; i < x.size(); i++)
    {
        res.data()[i] = (x.data()[i]) * d;
    }
    truncateMatrixXu(res);
    return res;
}

void Mat::residual(MatrixXi &a)
{
    for (int i = 0; i < a.size(); i++)
    {
        a.data()[i] = (a.data()[i] % MOD + MOD) % MOD;
    }
}

MatrixXu Mat::getFrom_pos(char *&p)
{
    vector<u64> val;
    int r = Constant::Util::char_to_int(p);
    int c = Constant::Util::char_to_int(p);
    val.resize(r * c);
    int l = r * c;
    for (int i = 0; i < l; i++)
    {
        val[i] = Constant::Util::char_to_u64(p);
    }
    return Mat::fromVector(val, r, c);
}

int Mat::toString_pos(char *p, int r, int c, vector<u64> val)
{
    Constant::Util::int_to_char(p, r);
    Constant::Util::int_to_char(p, c);
    int l = r * c;
    for (int i = 0; i < l; i++)
    {
        Constant::Util::u64_to_char(p, val[i]);
    }
    *p = 0;
    return 2 * 4 + r * c * 8;
}

vector<u64> Mat::toVector(MatrixXu &a)
{
    vector<u64> val;
    int r = a.rows();
    int c = a.cols();
    int l = r * c;
    val.resize(r * c);
    for (int i = 0; i < c; i++)
        for (int j = 0; j < r; j++)
        {
            val[(i * r + j) % l] = a(j, i);
        }
    return val;
}

MatrixXu Mat::fromVector(vector<u64> val, int r, int c)
{
    MatrixXu a(r, c);
    int l = r * c;

    for (int i = 0; i < c; i++)
        for (int j = 0; j < r; j++)
        {
            a(j, i) = val[(i * r + j) % l];
        }
    return a;
}

void Mat::Relu(MatrixXu &x)
{
    for (int i = 0; i < x.size(); i++)
    {
        if (x.data()[i] > UINT64_MAX / 2)
            x.data()[i] = 0;
    }
}

MatrixXu Mat::DRelu(MatrixXu &x)
{
    MatrixXu res(x.rows(), x.cols());
    for (int i = 0; i < x.size(); i++)
    {
        if (x.data()[i] >= UINT64_MAX / 2)
            res.data()[i] = 0;
        else
            res.data()[i] = 1;
    }
    return res;
}

void Mat::Sigmoid(MatrixXu &x)
{
    for (int i = 0; i < x.size(); i++)
    {
        if (Constant::Util::u64_to_double(x.data()[i]) <= -0.5)
            x.data()[i] = 0;
        else if (Constant::Util::u64_to_double(x.data()[i]) < 0.5)
            x.data()[i] += IE / 2;
        else
            x.data()[i] = IE;
    }
}

MatrixXu Mat::DSigmoid(MatrixXu &x)
{
    MatrixXu I(x.rows(), x.cols());
    I.fill(IE);
    Sigmoid(x);
    MatrixXu res = x.cwiseProduct(I - x);
    truncateMatrixXu(res);
    return res;
}

MatrixXb Mat::MatrixXb_xor(const MatrixXb &a, const MatrixXb &b)
{
    MatrixXb res(a.rows(), a.cols());
    for (int i = 0; i < a.size(); i++)
        res.data()[i] = a.data()[i] ^ b.data()[i];
    return res;
}
