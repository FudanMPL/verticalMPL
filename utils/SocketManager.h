#ifndef VERTICAL_SOCKETMANAGER_H
#define VERTICAL_SOCKETMANAGER_H

#include "SocketOnline.h"

extern int party;
extern SocketOnline *socket_io[M][M];

class SocketManager
{
public:
    static void init_windows_socket();
    static void exit_windows_socket();
    static void server_init(SOCK &sock, string ip, int port);
    static void client_init(SOCK &sock, string ip, int port);
    static void socket_close(SOCK sock);
    static SOCK accept_sock(SOCK sock);
    static void print_socket();
    static void print();
    class VMPL
    {
        string *ip;
        int *port;
        SOCK serv_sock;
        SOCK clnt_sock[M];
        SOCK sock;
        int epoch;

    public:
        VMPL();
        void init();
        void init(string *ip, int *port);
        void server();
        void client();
        void server_exit();
        void client_exit();
        void exit_all();
        void Send(MatrixXu *a);
        void Send(MatrixXb *b);
        void Send(MatrixXi *a);
        void Receive(MatrixXu *a);
        void Receive(MatrixXb *b);
        void Receive(MatrixXi *a);
    };
};

#endif // VERTICAL_SOCKETMANAGER_H
