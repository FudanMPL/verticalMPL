#include "NNModel.h"

NNModel::NNModel() {}

NNModel::NNModel(MatrixXu &train_data, MatrixXu &train_label, MatrixXu &test_data, MatrixXu &test_label)
{
    this->train_data = train_data;
    this->train_label = train_label;
    this->test_data = test_data;
    this->test_label = test_label;
    if (party == 0)
        col = D1;
    else
        col = D2;

    // the number of layers
    Layer_num = 4;
    // the number of nodes of each layer
    Node_num = {col, 128, 128, 1};
    // the activate function used in each layer
    activation = {"linear", "sigmoid", "sigmoid"};

    Layer.resize(Layer_num);
    dev.resize(Layer_num);
    error.resize(Layer_num);
    weight.resize(Layer_num - 1);
    delta.resize(Layer_num - 1);
    bias.resize(Layer_num - 1);
    bias_delta.resize(Layer_num - 1);

    default_random_engine generator;
    uniform_real_distribution<double> distribution(-0.1, 0.1);

    for (int i = 0; i < Layer_num; i++)
    {
        Layer[i].resize(B, Node_num[i]);
        dev[i].resize(B, Node_num[i]);
        error[i].resize(B, Node_num[i]);

        if (i != Layer_num - 1)
        {
            if (i == 0)
            {
                weight[i].resize(D, Node_num[i + 1]);
                delta[i].resize(D, Node_num[i + 1]);
            }
            else
            {
                weight[i].resize(Node_num[i], Node_num[i + 1]);
                delta[i].resize(Node_num[i], Node_num[i + 1]);
            }
            for (int j = 0; j < weight[i].size(); j++)
            {
                weight[i].data()[j] = Constant::Util::double_to_u64(distribution(generator));
            }
        }
    }

    for (int j = 0; j < bias.size(); j++)
    {
        bias_delta[j].resize(1, Node_num[j + 1]);
        bias[j].resize(1, Node_num[j + 1]);
        for (int k = 0; k < bias[j].size(); k++)
            bias[j].data()[k] = Constant::Util::double_to_u64(distribution(generator));
    }
    cout << "parameter intialized" << endl;
}

void NNModel::next_batch(MatrixXu &batch, int start, vector<int> &perm, MatrixXu &data)
{
    for (int i = 0; i < B; i++)
    {
        batch.row(i) = data.row(perm[start + i]);
    }
}

vector<int> NNModel::random_perm()
{
    vector<int> temp, perm;
    for (int i = 0; i < N; i++)
        temp.push_back(i);

    for (int i = 0; i < Ep; i++)
    {
        random_shuffle(temp.begin(), temp.end());           // random permutations of indices between 0 to N-1
        perm.insert(perm.end(), temp.begin(), temp.end()); // append the random permutated indicies
    }
    return perm;
}

void NNModel::Forward(int i)
{
    MatrixXu t(B, Node_num[i + 1]);

    if (i == 0)
    {
        MatrixXu t0 = weight[0].block(0, 0, D1, Node_num[1]);
        MatrixXu t1 = weight[0].block(D1, 0, D2, Node_num[1]);

        if (party == 0)
        {
            t = Functionalities::Multiply(Layer[0], B, Node_num[1], 0);
            t += Functionalities::Multiply(t1, B, Node_num[1], 1);
            t += Layer[0] * t0;
        }
        else
        {
            t = Functionalities::Multiply(t0, B, Node_num[1], 1);
            t += Functionalities::Multiply(Layer[0], B, Node_num[1], 0);
            t += Layer[0] * t1;
        }
    }
    else
    {
        t = Functionalities::ShareMultiply(Layer[i], weight[i]); // the share of layer[i] * weight[i]
    }
    Mat::truncateMatrixXu(t);
    for (int j = 0; j < t.rows(); j++)
    {
        t.row(j) += bias[i]; // add bias
    } 

    if (activation[i] == "relu")
    {
        Functionalities::Relu(t, dev[i + 1], Layer[i + 1]);
    }
    else if (activation[i] == "sigmoid")
    {
        Functionalities::DSigmoid(t, dev[i + 1], Layer[i + 1]);
    }
    else
    {
        Layer[i + 1] = t;
        dev[i + 1].setZero();
    }
}

void NNModel::Compute_Loss(MatrixXu &y_batch)
{
    error[Layer_num - 1] = Layer[Layer_num - 1] - y_batch;
}

void NNModel::Backward(int i)
{
    if (activation[i - 1] != "linear")
    {
        error[i] = Functionalities::SharecwiseProduct(error[i], dev[i]);
        if (activation[i - 1] == "sigmoid")
            Mat::truncateMatrixXu(error[i]);
    }
    if (i == 1)
    {
        MatrixXu L0_trans = Layer[0].transpose();
        if (party == 0)
        {
            MatrixXu t0 = Functionalities::Multiply(L0_trans, D1, Node_num[1], 0);
            MatrixXu t1 = Functionalities::Multiply(error[1], D2, Node_num[1], 1);
            delta[0].block(0, 0, D1, Node_num[1]) = t0 + L0_trans * error[1];
            delta[0].block(D1, 0, D2, Node_num[1]) = t1;
        }
        else
        {
            MatrixXu t0 = Functionalities::Multiply(error[1], D1, Node_num[1], 1);
            MatrixXu t1 = Functionalities::Multiply(L0_trans, D2, Node_num[1], 0);
            delta[0].block(0, 0, D1, Node_num[1]) = t0;
            delta[0].block(D1, 0, D2, Node_num[1]) = t1 + L0_trans * error[1];
        }
        Mat::truncateMatrixXu(delta[0]);
    }
    else
    {
        delta[i - 1] = Functionalities::ShareMultiply(Layer[i - 1].transpose(), error[i]);
        Mat::truncateMatrixXu(delta[i - 1]);
        bias_delta[i - 1] = error[i].colwise().sum();
        error[i - 1] = Functionalities::ShareMultiply(error[i], weight[i - 1].transpose());
        Mat::truncateMatrixXu(error[i - 1]);
    }
    Update(i - 1);
}

void NNModel::Update(int i)
{
    weight[i] -= Mat::constant_multiply(delta[i], A / B);
    bias[i] -= Mat::constant_multiply(bias_delta[i], A / B);
}
void NNModel::train_model()
{
    vector<int> perm = random_perm();
    MatrixXu x_batch(B, col);
    MatrixXu y_batch(B, 1);
    int start = 0;
    for (int i = 0; i < IT; i++)
    {
        next_batch(x_batch, start, perm, train_data);
        next_batch(y_batch, start, perm, train_label); // select mini batch
        Layer[0] = x_batch;
        start += B;
        for (int j = 0; j < Layer_num - 1; j++)
        {
            Forward(j);
        }
        Compute_Loss(y_batch);
        for (int j = Layer_num - 1; j > 0; j--)
        {
            Backward(j);
        }
        if (i % 100 == 0)
            inference(x_batch, y_batch);
        cout << i << " training done" << endl;
    }
}

void NNModel::inference(MatrixXu &data, MatrixXu &label)
{
    int d = data.rows();
    MatrixXu predicts(d, Node_num[1]);
    MatrixXu t0 = weight[0].block(0, 0, D1, Node_num[1]);
    MatrixXu t1 = weight[0].block(D1, 0, D2, Node_num[1]);
    if (party == 0)
    {
        predicts = Functionalities::Multiply(data, d, Node_num[1], 0);
        predicts += Functionalities::Multiply(t1, d, Node_num[1], 1);
        predicts += data * t0;
    }
    else
    {
        predicts = Functionalities::Multiply(t0, d, Node_num[1], 1);
        predicts += Functionalities::Multiply(data, d, Node_num[1], 0);
        predicts += data * t1;
    }
    Mat::truncateMatrixXu(predicts);

    for (int i = 1; i < Layer_num - 1; i++)
    {
        predicts = Functionalities::ShareMultiply(predicts, weight[i]); // the share of layer[i] * weight[i]
        Mat::truncateMatrixXu(predicts);
        for (int j = 0; j < d; j++)
        {
            predicts.row(j) += bias[i];
        } // add bias
        if (activation[i] == "relu")
        {
            predicts = Functionalities::Relu(predicts);
        }
        else if (activation[i] == "sigmoid")
        {
            predicts = Functionalities::Sigmoid(predicts);
        }
    }
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

void NNModel::test_model()
{
    int count = 0;
    MatrixXu predicts = test_data;
    for (int i = 0; i < Layer_num - 1; i++)
    {
        MatrixXu w = Functionalities::reveal(weight[i]);
        predicts *= w;
        MatrixXu b = Functionalities::reveal(bias[i]);
        Mat::truncateMatrixXu(predicts);
        for (int j = 0; j < predicts.rows(); j++)
            predicts.row(j) += b;
        if (activation[i] == "relu")
            Mat::Relu(predicts);
        else if (activation[i] == "sigmoid")
            Mat::Sigmoid(predicts);
    }

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