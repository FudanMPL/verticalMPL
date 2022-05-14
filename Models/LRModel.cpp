#include "LRModel.h"

LRModel::LRModel() {}

LRModel::LRModel(MatrixXu &train_data, MatrixXu &train_label, MatrixXu &test_data, MatrixXu &test_label)
{
    this->train_data = train_data;
    this->train_label = train_label;
    this->test_data = test_data;
    this->test_label = test_label;

    if (party == 0)
        col = D1;
    else
        col = D2;

    default_random_engine generator;
    uniform_real_distribution<double> distribution(-0.02, 0.02);
    for (int j = 0; j < w.size(); j++)
    {
        w.data()[j] = Constant::Util::double_to_u64(distribution(generator));
    }
    for (int j = 0; j < b.size(); j++)
    {
        b.data()[j] = Constant::Util::double_to_u64(distribution(generator));
    }
    delta.setZero();
    b_delta.setZero();
    activation = "sigmoid";
}

void LRModel::next_batch(MatrixXu &batch, int start, vector<int> &perm, MatrixXu &data)
{
    for (int i = 0; i < B; i++)
    {
        batch.row(i) = data.row(perm[start + i]);
    }
}

vector<int> LRModel::random_perm()
{
    vector<int> temp, perm;
    for (int i = 0; i < N; i++)
        temp.push_back(i);

    for (int i = 0; i < Ep; i++)
    {
        random_shuffle(temp.begin(), temp.end());          // random permutations of indices between 0 to N-1
        perm.insert(perm.end(), temp.begin(), temp.end()); // append the random permutated indicies
    }
    return perm;
}

void LRModel::Update()
{
    w -= Mat::constant_multiply(delta, A / B);
    b -= Mat::constant_multiply(b_delta, A / B);
}

void LRModel::linear_train()
{
    activation = "linear";
    vector<int> perm = random_perm();
    MatrixXu x_batch(B, col);
    MatrixXu y_batch(B, 1);

    int start = 0;

    for (int i = 0; i < IT; i++)
    {
        next_batch(x_batch, start, perm, train_data);
        next_batch(y_batch, start, perm, train_label); // select mini batch
        start += B;

        MatrixXu wx(B, 1), loss(B, 1); 
        wx.setZero();
        // Forward phase
        if (party == 0)
        {
            MatrixXu t1(D1, 1), t2(D2, 1);
            t1 = w.block(0, 0, D1, 1);
            t2 = w.block(D1, 0, D2, 1);

            wx += Functionalities::Multiply(x_batch, B, 1, 0);
            wx += Functionalities::Multiply(t2, B, 1, 1);
            wx += x_batch * t1;
        }
        else
        {
            MatrixXu t1(D1, 1), t2(D2, 1);
            t1 = w.block(0, 0, D1, 1);
            t2 = w.block(D1, 0, D2, 1);

            wx += Functionalities::Multiply(t1, B, 1, 1);
            wx += Functionalities::Multiply(x_batch, B, 1, 0);
            wx += x_batch * t2;
        }
        Mat::truncateMatrixXu(wx);
        for (int j = 0; j < wx.rows(); j++) // add bias
            wx.row(j) += b;

        loss = wx - y_batch;
        // Backward phase
        MatrixXu t[M];
        MatrixXu x_trans = x_batch.transpose();

        if (party == 0)
        {
            t[0] = Functionalities::Multiply(x_trans, D1, 1, 0);
            t[1] = Functionalities::Multiply(loss, D2, 1, 1);
            delta.block(0, 0, D1, 1) = t[0] + x_trans * loss;
            delta.block(D1, 0, D2, 1) = t[1];
        }
        else
        {
            t[0] = Functionalities::Multiply(loss, D1, 1, 1);
            t[1] = Functionalities::Multiply(x_trans, D2, 1, 0);
            delta.block(0, 0, D1, 1) = t[0];
            delta.block(D1, 0, D2, 1) = t[1] + x_trans * loss;
        }

        Mat::truncateMatrixXu(delta);
        b_delta = loss.colwise().sum();
        Update();

        if (i % 10 == 0)
        inference(x_batch, y_batch);
        cout << i + 1 << " training done" << endl;
    }
}

void LRModel::logistic_train()
{
    vector<int> perm = random_perm();
    MatrixXu x_batch(B, col);
    MatrixXu y_batch(B, 1);

    int start = 0;

    for (int i = 0; i < IT; i++)
    {
        next_batch(x_batch, start, perm, train_data);
        next_batch(y_batch, start, perm, train_label); // select mini batch

        start += B;
        MatrixXu wx(B, 1), loss(B, 1);
        wx.setZero();

        // Forward phase
        if (party == 0)
        {
            MatrixXu t1(D1, 1), t2(D2, 1);
            t1 = w.block(0, 0, D1, 1);
            t2 = w.block(D1, 0, D2, 1);

            wx += Functionalities::Multiply(x_batch, B, 1, 0);
            wx += Functionalities::Multiply(t2, B, 1, 1);
            wx += x_batch * t1;
        }
        else
        {
            MatrixXu t1(D1, 1), t2(D2, 1);
            t1 = w.block(0, 0, D1, 1);
            t2 = w.block(D1, 0, D2, 1);

            wx += Functionalities::Multiply(t1, B, 1, 1);
            wx += Functionalities::Multiply(x_batch, B, 1, 0);
            wx += x_batch * t2;
        }
        Mat::truncateMatrixXu(wx);

        for (int j = 0; j < wx.rows(); j++)
            wx.row(j) += b;

        if (activation == "relu")
        {
            MatrixXu drelu(B, 1), relu(B, 1);
            Functionalities::Relu(wx, drelu, relu);
            loss = Functionalities::SharecwiseProduct(relu - y_batch, drelu);
        }
        else if (activation == "sigmoid")
        {
            loss = Functionalities::Sigmoid(wx) - y_batch;
        }

        // Backward phase
        MatrixXu t[M];
        MatrixXu x_trans = x_batch.transpose();

        if (party == 0)
        {
            t[0] = Functionalities::Multiply(x_trans, D1, 1, 0);
            t[1] = Functionalities::Multiply(loss, D2, 1, 1);
            delta.block(0, 0, D1, 1) = t[0] + x_trans * loss;
            delta.block(D1, 0, D2, 1) = t[1];
        }
        else
        {
            t[0] = Functionalities::Multiply(loss, D1, 1, 1);
            t[1] = Functionalities::Multiply(x_trans, D2, 1, 0);
            delta.block(0, 0, D1, 1) = t[0];
            delta.block(D1, 0, D2, 1) = t[1] + x_trans * loss;
        }
        Mat::truncateMatrixXu(delta);
        b_delta = loss.colwise().sum();
        Update();

        if (i % 10 == 0)
        inference(x_batch, y_batch);
        cout << i + 1 << " training done" << endl;
    }
}

void LRModel::inference(MatrixXu &data, MatrixXu &label)
{
    int d = data.rows();
    MatrixXu predicts(d, 1);
    predicts.setZero();
    if (party == 0)
    {
        MatrixXu t1(D1, 1), t2(D2, 1);
        t1 = w.block(0, 0, D1, 1);
        t2 = w.block(D1, 0, D2, 1);

        predicts += Functionalities::Multiply(data, d, 1, 0);
        predicts += Functionalities::Multiply(t2, d, 1, 1);
        predicts += data * t1;
    }
    else
    {
        MatrixXu t1(D1, 1), t2(D2, 1);
        t1 = w.block(0, 0, D1, 1);
        t2 = w.block(D1, 0, D2, 1);

        predicts += Functionalities::Multiply(t1, d, 1, 1);
        predicts += Functionalities::Multiply(data, d, 1, 0);
        predicts += data * t2;
    }
    Mat::truncateMatrixXu(predicts);
    for (int j = 0; j < d; j++) // add bias
        predicts.row(j) += b;

    if (activation == "relu")
        predicts = Functionalities::Relu(predicts);
    else if (activation == "sigmoid")
        predicts = Functionalities::Sigmoid(predicts);

    predicts = Functionalities::reveal(predicts);
    if (party == 1)
    {
        int count = 0;
        for (int i = 0; i < d; i++)
        {
            if (Constant::Util::u64_to_double(predicts(i, 0)) >= 0.5 && label(i, 0) == IE)
                count++;
            else if (Constant::Util::u64_to_double(predicts(i, 0)) < 0.5 && label(i, 0) == 0)
                count++;

            cout << "predict: " << Constant::Util::u64_to_double(predicts(i, 0)) << " "
                 << "label: " << label(i, 0) / IE << endl;
        }
        cout << "accuracy: " << count * 1.0 / d * 100 << "%" << endl;
    }
}

void LRModel::test_model()
{
    int count = 0;
    w = Functionalities::reveal(w);
    b = Functionalities::reveal(b);
    MatrixXu predicts = test_data * w;
    Mat::truncateMatrixXu(predicts);
    for (int j = 0; j < predicts.rows(); j++)
        predicts.row(j) += b;
    if (activation == "relu")
        Mat::Relu(predicts);
    else if (activation == "sigmoid")
        Mat::Sigmoid(predicts);

    for (int i = 0; i < testN; i++)
    {
        if (Constant::Util::u64_to_double(predicts(i, 0)) >= 0.5 && test_label(i, 0) == IE)
            count++;
        else if (Constant::Util::u64_to_double(predicts(i, 0)) < 0.5 && test_label(i, 0) == 0)
            count++;

        cout << "predict: " << Constant::Util::u64_to_double(predicts(i, 0)) << " "
             << "label: " << test_label(i, 0) / IE << endl;
    }
    cout << "accuracy: " << count * 1.0 / testN * 100 << "%" << endl;
}