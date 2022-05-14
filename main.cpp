#include "Models/LRModel.h"
#include "Models/NNModel.h"
#include "utils/IOManager.h"
using namespace std;

int party;
SocketManager::VMPL tel;

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        DBGprint("Please enter party index:\n");
        scanf("%d", &party);
    }
    else
    {
        party = argv[1][0] - '0';
    }
    DBGprint("party index: %d\n", party);
    tel.init();

    ifstream F1, F2;
    F1.open("../data/mnist/mnist_train_party" + to_string(party) + ".csv");
    F2.open("../data/mnist/mnist_test_.csv");
    IOManager::init(F1, F2);
    IOManager::norm(256); //normaliztion
    F1.close();
    F2.close();
    LRModel *LR = new LRModel(IOManager::train_data, IOManager::train_label, IOManager::test_data, IOManager::test_label);
    NNModel *NN = new NNModel(IOManager::train_data, IOManager::train_label, IOManager::test_data, IOManager::test_label);
    int t1 = 1;
    Constant::Clock c1 = Constant::Clock(t1);

    /****linear regression****/
    LR->linear_train(); 

    /****logistic regression****/        
    // LR->logistic_train();   

    /****neural network****/ 
    // NN->train_model();   

    cout << "training time : ";
    c1.print();
    
    /****test model****/ 
    // LR->test_model();
    // NN->test_model();
    // c1.print();
}