#include "Functionalities.h"

MatrixXu Functionalities::Multiply(const MatrixXu &a, int rows, int cols, int pos)
{
    assert(pos == 1 || pos == 0);
    if (pos == 0)
    {
        if (a.cols() % 2 == 0)
        {
            return Multiply_Matrix(a, rows, cols, pos);
        }
        else if (a.cols() == 1)
        {
            MatrixXu r1(rows, 1), r2;
            PreCompute::getRandomMatrix(r1);
            tel.Send(&r1);
            tel.Receive(&r2);
            return ShareMultiply_Vector(a - r1, r2);
        }
        else
        {
            MatrixXu x = a.block(0, 0, rows, a.cols() - 1);
            MatrixXu r1(rows, 1), r2;
            PreCompute::getRandomMatrix(r1);
            MatrixXu y = a.col(a.cols() - 1) - r1;
            tel.Send(&r1);
            tel.Receive(&r2);

            return Multiply_Matrix(x, rows, cols, pos) + ShareMultiply_Vector(y, r2);
        }
    }
    else
    {
        if (a.rows() % 2 == 0)
        {
            return Multiply_Matrix(a, rows, cols, pos);
        }
        else if (a.rows() == 1)
        {
            MatrixXu r2(1, cols), r1;
            PreCompute::getRandomMatrix(r2);
            tel.Receive(&r1);
            tel.Send(&r2);
            return ShareMultiply_Vector(r1, a - r2);
        }
        else
        {
            MatrixXu x = a.block(0, 0, a.rows() - 1, a.cols());
            MatrixXu r2(1, cols), r1;
            PreCompute::getRandomMatrix(r2);
            MatrixXu y = a.row(a.rows() - 1) - r2;
            tel.Receive(&r1);
            tel.Send(&r2);
            return Multiply_Matrix(x, rows, cols, pos) + ShareMultiply_Vector(r1, y);
        }
    }
}

MatrixXu Functionalities::ShareMultiply(const MatrixXu &x, const MatrixXu &y)
{
    assert(x.cols() == y.rows());
    if (x.cols() % 2 == 0)
        return ShareMultiply_Matrix(x, y);
    else if (x.cols() == 1)
        return ShareMultiply_Vector(x, y);
    else
    {
        MatrixXu x1 = x.block(0, 0, x.rows(), x.cols() - 1);
        MatrixXu x2 = x.col(x.cols() - 1);
        MatrixXu y1 = y.block(0, 0, y.rows() - 1, y.cols());
        MatrixXu y2 = y.row(y.rows() - 1);
        return ShareMultiply_Matrix(x1, y1) + ShareMultiply_Vector(x2, y2);
    }
}

MatrixXu Functionalities::Multiply_Matrix(const MatrixXu &x, int rows, int cols, int pos)
{
    if (pos == 0)
    {
        int c = x.cols();
        MatrixXu a(rows, c), a2(rows, c / 2), b1(c / 2, cols), y_b;
        PreCompute::getRandomnessforMultiply(a, a2, b1, pos);
        MatrixXu x_a = x + a;
        tel.Send(&x_a);
        tel.Receive(&y_b);
        return (x + 2 * a) * y_b + a2 * b1;
    }
    else
    {
        int r = x.rows();
        MatrixXu b(r, cols), b2(r / 2, cols), a1(rows, r / 2), y_a;
        PreCompute::getRandomnessforMultiply(b, b2, a1, pos);
        MatrixXu x_b = b - x;
        tel.Receive(&y_a);
        tel.Send(&x_b);
        return y_a * (2 * x - b) + a1 * b2;
    }
}

MatrixXu Functionalities::ShareMultiply_Vector(const MatrixXu &x, const MatrixXu &y) // multiplicaiton of the last column & row
{
    assert(x.cols() == y.rows());
    int row = x.rows();
    int col = y.cols();

    if (party == 0)
    {
        MatrixXu a(row, 1), c(1, col), ac2(row, col), bd1(row, col), u(1, row + col), v;
        PreCompute::getRandomnessforShareMultiplyV(a, c, ac2, bd1, 0);
        MatrixXu x_a = (x + a).transpose();
        MatrixXu y_c = y + c;
        u << x_a, y_c;
        tel.Send(&u);
        tel.Receive(&v);

        return (x + 2 * a) * v.block(0, row, 1, col) + v.block(0, 0, 1, row).transpose() * (y + 2 * c) + ac2.cwiseProduct(bd1) + x * y;
    }
    else
    {
        MatrixXu d(row, 1), b(1, col), bd2(row, col), ac1(row, col), v(1, row + col), u;
        PreCompute::getRandomnessforShareMultiplyV(d, b, bd2, ac1, 1);
        MatrixXu d_x = (d - x).transpose();
        MatrixXu b_y = b - y;
        v << d_x, b_y;
        tel.Receive(&u);
        tel.Send(&v);

        return (2 * x - d) * u.block(0, row, 1, col) + u.block(0, 0, 1, row).transpose() * (2 * y - b) + ac1.cwiseProduct(bd2) + x * y;
    }
}

MatrixXu Functionalities::ShareMultiply_Matrix(const MatrixXu &x, const MatrixXu &y)
{
    int xr = x.rows(), xc = x.cols(), yr = y.rows(), yc = y.cols();

    MatrixXu a(xr, xc), b(yr, yc), a2(xr, xc / 2), b2(yr / 2, yc), a1(xr, xc / 2), b1(yr / 2, yc); // a1 and b2 are the random matrices related to the other party
    PreCompute::getRandomnessforShareMultiplyM(a, b, a2, b2, a1, b1);
    MatrixXu p(xr, xc / 2), p1(xr, xc / 2), q(yr / 2, yc), q1(yr / 2, yc);
    MatrixXu u(xr + yc, xc), v;
    u << x + a,
        (b - y).transpose();

    if (party == 0)
    {
        tel.Send(&u);
        tel.Receive(&v);
    }
    else
    {
        tel.Receive(&v);
        tel.Send(&u);
    }
    MatrixXu x2 = v.topLeftCorner(xr, xc);
    MatrixXu y2 = v.bottomLeftCorner(yc, yr).transpose();

    return x * y + (x + 2 * a) * y2 + x2 * (2 * y - b) + a2 * b1 + a1 * b2;
}

void Functionalities::SharecwiseProduct(const vector<MatrixXi> &v, const MatrixXi &w1, vector<MatrixXi> &u)
{
    // compute u[i] = cwiseproduct(v[i] , w)
    assert(w1.cols() == 1);
    
    int size = v.size();
    int rows = v[0].rows();
    int cols = v[0].cols();
    MatrixXi w(rows, cols);
    for (int i = 0; i < cols; i++)
        w.col(i) = w1.col(0);
    if (party == 0)
    {
        vector<MatrixXi> a(size), ac2(size), bd1(size);
        MatrixXi c(rows, cols), p(rows, cols * (size + 1)), q, m(rows, cols), n(rows, cols);
        PreCompute::getRandomnessforSharecwiseProduct(a, c, ac2, bd1, rows, cols, 0);
        vector<MatrixXi> p1(size);
        for (int i = 0; i < size; i++)
        {
            p.block(0, i * cols, rows, cols) = a[i] + v[i];
            p1[i] = 2 * a[i] + v[i];
            Mat::residual(p1[i]);
        }
        p.topRightCorner(rows, cols) = w + c;
        Mat::residual(p);
        tel.Send(&p);
        tel.Receive(&q);
        m = w + 2 * c;
        Mat::residual(m);
        n = q.topRightCorner(rows, cols);
        for (int i = 0; i < size; i++)
        {
            u[i] = p1[i].cwiseProduct(n) + q.block(0, i * cols, rows, cols).cwiseProduct(m) + ac2[i].cwiseProduct(bd1[i]) + v[i].cwiseProduct(w);
            Mat::residual(u[i]);
        }
    }
    else
    {
        vector<MatrixXi> d(size), bd2(size), ac1(size);
        MatrixXi b(rows, cols);
        PreCompute::getRandomnessforSharecwiseProduct(d, b, bd2, ac1, rows, cols, 1);
        MatrixXi q(rows, cols * (size + 1)), p, n(rows, cols), m(rows, cols);
        vector<MatrixXi> q1(size);
        for (int i = 0; i < size; i++)
        {
            q.block(0, i * cols, rows, cols) = d[i] - v[i];
            q1[i] = 2 * v[i] - d[i];
            Mat::residual(q1[i]);
        }
        q.topRightCorner(rows, cols) = b - w;
        Mat::residual(q);
        tel.Receive(&p);
        tel.Send(&q);
        n = 2 * w - b;
        Mat::residual(n);
        m = p.topRightCorner(rows, cols);
        for (int i = 0; i < size; i++)
        {
            u[i] = n.cwiseProduct(p.block(0, i * cols, rows, cols)) + q1[i].cwiseProduct(m) + bd2[i].cwiseProduct(ac1[i]) + v[i].cwiseProduct(w);
            Mat::residual(u[i]);
        }
    }
}

MatrixXu Functionalities::SharecwiseProduct(const MatrixXu &x, const MatrixXu &y)
{
    int rows = x.rows(), cols = x.cols();
    assert(x.rows() == y.rows() && x.cols() == y.cols());

    if (party == 0)
    {
        MatrixXu a(rows, cols), c(rows, cols), bd1(rows, cols);
        PreCompute::getRandomnessforSharecwiseProduct(a, c, bd1, 0);
        MatrixXu p(rows, 2 * cols), q, q1(rows, cols), q2(rows, cols), q3(rows, cols);
        p << x + a, y + c;
        tel.Send(&p);
        tel.Receive(&q);
        q1 = q.block(0, 0, rows, cols);
        q2 = q.block(0, cols, rows, cols);

        MatrixXu result = (x + 2 * a).cwiseProduct(q1) + (y + 2 * c).cwiseProduct(q2) + (a + 2 * c).cwiseProduct(bd1);
        return result + x.cwiseProduct(y);
    }
    else
    {
        MatrixXu d(rows, cols), b(rows, cols), ac1(rows, cols);
        PreCompute::getRandomnessforSharecwiseProduct(d, b, ac1, 1);
        MatrixXu p, q(rows, 2 * cols), p1(rows, cols), p2(rows, cols);
        q << b - y, d - x;
        tel.Receive(&p);
        tel.Send(&q);
        p1 = p.block(0, 0, rows, cols);
        p2 = p.block(0, cols, rows, cols);

        MatrixXu result = (2 * y - b).cwiseProduct(p1) + (2 * x - d).cwiseProduct(p2) + (d - 2 * b).cwiseProduct(ac1);
        return result + x.cwiseProduct(y);
    }
}

MatrixXi Functionalities::SharecwiseProduct(const MatrixXi &x, const MatrixXi &y)
{
    int rows = x.rows(), cols = x.cols();
    assert(x.rows() == y.rows() && x.cols() == y.cols());

    if (party == 0)
    {
        MatrixXi a(rows, cols), c(rows, cols), bd1(rows, cols);
        PreCompute::getRandomnessforSharecwiseProduct(a, c, bd1, 0);
        MatrixXi p(rows, 2 * cols), q, q1(rows, cols), q2(rows, cols), q3(rows, cols);
        p << x + a, y + c;
        Mat::residual(p);
        tel.Send(&p);
        tel.Receive(&q);
        q1 = q.block(0, 0, rows, cols);
        q2 = q.block(0, cols, rows, cols);

        MatrixXi result = (x + 2 * a).cwiseProduct(q1) + (y + 2 * c).cwiseProduct(q2) + (a + 2 * c).cwiseProduct(bd1) + x.cwiseProduct(y);
        Mat::residual(result);
        return result;
    }
    else
    {
        MatrixXi d(rows, cols), b(rows, cols), ac1(rows, cols);
        PreCompute::getRandomnessforSharecwiseProduct(d, b,ac1, 1);
        MatrixXi p, q(rows, 2 * cols), p1(rows, cols), p2(rows, cols);
        q << b - y, d - x;
        Mat::residual(q);
        tel.Receive(&p);
        tel.Send(&q);
        p1 = p.block(0, 0, rows, cols);
        p2 = p.block(0, cols, rows, cols);

        MatrixXi result = (2 * y - b).cwiseProduct(p1) + (2 * x - d).cwiseProduct(p2) + (d - 2 * b).cwiseProduct(ac1) + x.cwiseProduct(y);
        Mat::residual(result);
        return result;
    }
}

MatrixXu Functionalities::B2A(const MatrixXb &b)
{
    int r = b.rows(), c = b.cols();
    MatrixXb randomBit(r, c), e(r, c), e_rec;
    MatrixXu d(r, c), randomBitShare(r, c);
    PreCompute::getRandomBitShare(randomBit, randomBitShare);
    e = Mat::MatrixXb_xor(b, randomBit);
    e_rec = reveal(e); // reveal b XOR c
    for (int i = 0; i < e.size(); i++)
    {
        if (e_rec.data()[i])

            d.data()[i] = party - randomBitShare.data()[i];
        else
            d.data()[i] = randomBitShare.data()[i];
    }
    return d;
}

MatrixXu Functionalities::selectshare(MatrixXu &a, MatrixXb &b) // 0 or a
{
    MatrixXu d = B2A(b);
    return SharecwiseProduct(a, d);
}

MatrixXb Functionalities::privateCompare(const MatrixXi &xi, const MatrixXu &r, const MatrixXb &b, const MatrixXi &b_share) // from falcon
{
    // xi is the addictive sharing of bits of x in the small field
    // b_share is the addictive sharing of b in the small field, the column of b and b_share is 1
    int rows = r.rows(), cols = r.cols();
    MatrixXb b_(rows, cols);
    MatrixXi b_minus(rows, 1), d(rows, cols);

    vector<MatrixXb> dec_r(cols);
    vector<MatrixXi> v(cols), u(cols), w(cols), c(cols);
    for (int k = 0; k < cols; k++)
    {
        dec_r[k].resize(rows, P);
        v[k].resize(rows, P);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < P; j++)
            {
                dec_r[k](i, j) = static_cast<bool>((r(i, k) >> (P - j - 1)) & (u64)1); // bit decomposition of r
                v[k](i, j) = xi(i, j) - party * dec_r[k](i, j);
            }
            if (k == 0)
            {
                b_minus(i, 0) = party - 2 * b_share(i, 0); // compute (-1)^b
            }
        }
        Mat::residual(v[k]);
    }
    Mat::residual(b_minus);
    SharecwiseProduct(v, b_minus, u);

    for (int k = 0; k < cols; k++)
    {
        w[k].resize(rows, P);
        c[k].resize(rows, P);
        for (int i = 0; i < rows; i++)
        {
            int sum = 0;
            for (int j = 0; j < P; j++)
            {
                w[k](i, j) = xi(i, j) + party * dec_r[k](i, j) - 2 * dec_r[k](i, j) * xi(i, j); // compute x xor r
                if (j != 0)
                    sum += w[k](i, j - 1);
                c[k](i, j) = u[k](i, j) + party + sum;
            }
        }
        Mat::residual(c[k]);
        int t = P;
        MatrixXi res = c[k];
        while (t != 1)
        {
            t /= 2;
            MatrixXi c1 = res.topLeftCorner(rows, t);
            MatrixXi c2 = res.topRightCorner(rows, t);
            res.resize(rows, t);
            res = SharecwiseProduct(c1, c2);
            Mat::residual(res);
        }
        d.col(k) = res;
    }
    MatrixXi d_rec = reveal(d);
    bool p = party;
    for (int k = 0; k < cols; k++)
    {
        for (int i = 0; i < rows; i++)
        {
            b_(i, k) = p ^ b(i, 0);
            if (d_rec(i, k) == 0)
            {
                b_(i, k) = 0 ^ b(i, 0);
            }
        }
    }
    return b_;
}

MatrixXb Functionalities::wrap(MatrixXi &xi, MatrixXu1 &x, vector<int> alpha, const MatrixXu &a)
{
    // xi for the additive sharing of every bit of x in the small field Z/37
    int rows = a.rows(), cols = a.cols();
    MatrixXu r_rec, r(rows, cols);
    MatrixXi beta(rows, cols), delta(rows, cols);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            r(i, j) = a(i, j) + x(i, 0);

            if (r(i, j) > a(i, j))
                beta(i, j) = 0;
            else
                beta(i, j) = 1;
        }
    }
    r_rec = reveal(r);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (r_rec(i, j) > r(i, j))
                delta(i, j) = 0;
            else
                delta(i, j) = 1;
            r_rec(i, j)++;
        }
    }

    MatrixXb b(rows, 1), result(rows, cols);
    MatrixXi b_share(rows, 1);
    PreCompute::getRandomBitShare(b, b_share);
    MatrixXb eta = privateCompare(xi, r_rec, b, b_share);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            result(i, j) = (beta(i, j) + delta(i, j) * party - eta(i, j) - party * alpha[i] + 2) % 2;
        }
    }

    return result;
}

MatrixXb Functionalities::MSB(const MatrixXu &a)
{
    int rows = a.rows(), cols = a.cols();

    MatrixXi xi(rows, P);
    MatrixXu1 x(rows, 1);
    vector<int> w(rows); // wrap of x
    MatrixXb msb(rows, cols), result(rows, cols);

    PreCompute::getRandomnessforWrap(xi, x, w);

    u64 m = 1ll << 63;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            msb(i, j) = (bool)(m & a(i, j));
        }
    }
    result = Mat::MatrixXb_xor(msb, wrap(xi, x, w, 2 * a));

    return result;
}

MatrixXu Functionalities::DRelu(const MatrixXu &a)
{
    int rows = a.rows(), cols = a.cols();
    MatrixXb t(rows, cols), result(rows, cols);
    t.fill((bool)party);
    result = Mat::MatrixXb_xor(t, MSB(a));
    return B2A(result);
}

MatrixXu Functionalities::Relu(MatrixXu &wx)
{
    MatrixXu b = DRelu(wx);
    return SharecwiseProduct(wx, b);
}

void Functionalities::Relu(MatrixXu &wx, MatrixXu &dr, MatrixXu &r) // dr = DeLU(wx), r = ReLu(wx)
{
    dr = DRelu(wx);
    r = SharecwiseProduct(dr, wx);
}

void Functionalities::DSigmoid(MatrixXu &wx, MatrixXu &ds, MatrixXu &s)
{
    s = Sigmoid(wx);
    MatrixXu I(wx.rows(), wx.cols());
    I.fill(IE);
    ds = SharecwiseProduct(s, I - s);
    Mat::truncateMatrixXu(ds);
}

MatrixXu Functionalities::Sigmoid(MatrixXu &wx)
{
    int rows = wx.rows(), cols = wx.cols();
    MatrixXb b1(rows, cols), b2(rows, cols), t1(rows, cols);
    MatrixXu t2(rows, cols);
    t1.fill((bool)party);
    t2.fill(party * IE / 2);
    b1 = MSB(wx + t2); // MSB of wx+1/2
    b2 = MSB(wx - t2); // MSB of wx-1/2
    MatrixXu a1 = B2A(Mat::MatrixXb_xor(t1, b1));
    MatrixXu a2 = B2A(b2);
    MatrixXu a3 = B2A(Mat::MatrixXb_xor(t1, b2));
    MatrixXu result = SharecwiseProduct(a1, wx + t2);
    result = SharecwiseProduct(a2, result) + a3 * IE;

    return result;
}

MatrixXu Functionalities::reveal(const MatrixXu &a, int target)
{
    MatrixXu a_;
    if (party == target)
    {
        tel.Receive(&a_);
        return a + a_;
    }
    else
    {
        a_ = a;
        tel.Send(&a_);
        return a_;
    }
}

MatrixXu Functionalities::reveal(const MatrixXu &a)
{
    MatrixXu a1 = a, a2;
    if (party == 0)
    {
        tel.Receive(&a2);
        tel.Send(&a1);
    }
    else
    {
        tel.Send(&a1);
        tel.Receive(&a2);
    }
    return a + a2;
}

MatrixXb Functionalities::reveal(const MatrixXb &b)
{
    MatrixXb b1 = b, b2;
    if (party == 0)
    {
        tel.Receive(&b2);
        tel.Send(&b1);
    }
    else
    {
        tel.Send(&b1);
        tel.Receive(&b2);
    }
    return Mat::MatrixXb_xor(b1, b2);
}

MatrixXi Functionalities::reveal(const MatrixXi &a)
{
    // reveal secret in the small field Z/37
    MatrixXi a1 = a, a2;
    if (party == 0)
    {
        tel.Receive(&a2);
        tel.Send(&a1);
    }
    else
    {
        tel.Send(&a1);
        tel.Receive(&a2);
    }
    a1 += a2;
    Mat::residual(a1);
    return a1;
}