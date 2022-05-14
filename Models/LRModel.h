#ifndef VERTICALMPL_LRMODEL_H
#define VERTICALMPL_LRMODEL_H

#include "../SCParty/Functionalities.h"
#include "../utils/IOManager.h"
#include <string>

class LRModel
{
public:
    MatrixXu train_data;
    MatrixXu train_label;
    MatrixXu test_data;
    MatrixXu test_label;
    
    int col;
    MatrixXu w = MatrixXu(D, 1);
    MatrixXu delta = MatrixXu(D, 1);
    MatrixXu b = MatrixXu(1, 1);
    MatrixXu b_delta = MatrixXu(1, 1);
    string activation;

    LRModel();
    LRModel(MatrixXu &train_data, MatrixXu &train_label, MatrixXu &test_data, MatrixXu &test_label);
    void linear_train();
    void logistic_train();
    vector<int> random_perm();
    void next_batch(MatrixXu &batch, int start, vector<int> &perm, MatrixXu &data);
    void Update();
    void inference(MatrixXu &data, MatrixXu &label);
    void test_model();
};

#endif