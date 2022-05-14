#include "IOManager.h"
#include <string>

MatrixXu IOManager::train_data;
MatrixXu IOManager::train_label;
MatrixXu IOManager::test_data;
MatrixXu IOManager::test_label;

int IOManager::getsize(ifstream &in)
{
    string s;
    int count = 0;
    getline(in, s);
    for (int i = 0; i < s.length(); i++)
    {
        if (s[i] == ',')
            count++;
    }
    in.seekg(0, ios::beg);
    return count + 1;
}

void IOManager::init(ifstream &F1, ifstream &F2)
{
    load_train_data(F1);
    load_test_data(F2);
}

void IOManager::norm(int max)
{
    for (int i = 0; i < train_data.size(); i++)
    {
        train_data.data()[i] /= max;
    }
    for (int i = 0; i < test_data.size(); i++)
    {
        test_data.data()[i] /= max;
    }
}

void IOManager::load_train_data(ifstream &in)
{
    int i = 0;
    int num_col = IOManager::getsize(in);
    int num_row = N;
    if (party == 1)
        train_data.resize(num_row, num_col - 1);
    else
        train_data.resize(num_row, num_col);
    train_label.resize(num_row, 1);
    while (in)
    {
        string s;
        if (!getline(in, s))
            break;
        char *ch = const_cast<char *>(s.c_str());
        int temp;
        for (int j = 0; j < num_col; j++)
        {
            temp = Constant::Util::getint(ch);

            if (party == 1 && j == num_col - 1)
            {
                if (temp > 0)
                    train_label(i, 0) = IE;
                else
                    train_label(i, 0) = 0;
            }
            else
                train_data(i, j) = temp * IE;
        }
        if (party == 0)
            train_label(i, 0) = 0;
        i++;
        if (i >= num_row)
            break;
    }
    cout << train_data.rows() << " " << train_data.cols() << endl;
}

void IOManager::load_test_data(ifstream &in)
{
    int i = 0;
    int num_col = IOManager::getsize(in);
    int num_row = testN;
    test_data.resize(num_row, num_col - 1);
    test_label.resize(num_row, 1);
    while (in)
    {
        string s;
        if (!getline(in, s))
            break;
        char *ch = const_cast<char *>(s.c_str());

        int temp;

        for (int j = 0; j < num_col; j++)
        {
            temp = Constant::Util::getint(ch);
            if (j != num_col - 1)
            {
                test_data(i, j) = temp * IE;
            }
            else
            {
                if (temp > 0)
                    test_label(i, 0) = IE;
                else
                    test_label(i, 0) = 0;
            }
        }
        i++;
        if (i >= num_row)
            break;
    }
}
