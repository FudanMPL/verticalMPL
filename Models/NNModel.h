#ifndef VERTICALMPL_NMODEL_H
#define VERTICALMPL_NNMODEL_H

#include "../SCParty/Functionalities.h"
#include "../utils/IOManager.h"
#include <string>

class NNModel
{
public:
    MatrixXu train_data;
    MatrixXu train_label;
    MatrixXu test_data;
    MatrixXu test_label;
    
    int col;
    int Layer_num;
    vector<int> Node_num;
    vector<MatrixXu> Layer;
    vector<MatrixXu> weight;
    vector<MatrixXu> bias;
    vector<MatrixXu> bias_delta;
    vector<MatrixXu> dev;
    vector<MatrixXu> delta;
    vector<MatrixXu> error;
    vector<string> activation;

    NNModel();
    NNModel(MatrixXu &train_data, MatrixXu &train_label, MatrixXu &test_data, MatrixXu &test_label);
    vector<int> random_perm();
    void next_batch(MatrixXu &batch, int start, vector<int> &perm, MatrixXu &data);
    void Compute_Loss(MatrixXu &y_batch);
    void Forward(int i);
    void Backward(int i);
    void Update(int i);
    void train_model();
    void inference(MatrixXu &data, MatrixXu &label);
    void test_model();
};

#endif