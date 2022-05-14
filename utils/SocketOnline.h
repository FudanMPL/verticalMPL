#ifndef VERTICAL_SOCKETONLINE_H
#define VERTICAL_SOCKETONLINE_H

#include "Mat.h"

#ifdef UNIX_PLATFORM
typedef int SOCK;
#else
typedef SOCKET SOCK;
#endif

class SocketOnline
{
public:
    int id;
    SOCK sock;
    char *buffer;
    char *header;
    ll send_num;
    ll recv_num;
    SocketOnline();
    ~SocketOnline();
    static void test();
    SocketOnline(int id, SOCK sock);
    SocketOnline &operator=(const SocketOnline &u);
    void init(int id, SOCK sock);
    void init();
    void reset();
    int send_message(SOCK sock, char *p, int l);
    int send_message_n(SOCK sock, char *p, int l);
    int recv_message(SOCK sock, char *p, int l);
    int recv_message_n(SOCK sock, char *p, int l);
    void send_bits(MatrixXb *b);
    void recv_bits(MatrixXb &b);
    void send_message(MatrixXu &a);
    void send_message(MatrixXu *a);
    void send_message(MatrixXi *a);
    void recv_message(MatrixXi &a);
    MatrixXu recv_message();
    void recv_message(MatrixXu &a);
    void push(const MatrixXu &a);
    void print();
};

#endif // VERTICAL_SOCKETONLINE_H
