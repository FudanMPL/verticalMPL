#ifndef VERTICALMPL_IOMANAGER_H
#define VERTICALMPL_IOMANAGER_H

#include "Mat.h"

extern int party;

class IOManager
{
public:
    static MatrixXu train_data, train_label;
    static MatrixXu test_data, test_label;
    static int getsize(ifstream &in); 
    static void init(ifstream &F1, ifstream &F2);
    static void load_train_data(ifstream &in);
    static void load_test_data(ifstream &in);
    static void norm(int max);
};
#endif // VERTICAL_IOMANAGER_H