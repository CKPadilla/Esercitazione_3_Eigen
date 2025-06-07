/*
Nome, cognome e matricola: Cael Kyler Padilla | 307931
Esercitazione_3_Eigen
*/

#include <iostream>
#include "Eigen/Eigen"

using namespace std;
using namespace Eigen;

/* 
// Ho 3 matrici A e 3 matrici b, entrambi di dimensione costante
// Obiettivo: costruire funzioni che calcoli il risultato di A*b con fattorizzazione PALU e QR
*/

/*
//Funzione: risolvi_conPalu
//Inputs:
const MatrixXd& A : matrice A di dimensione "X", che va fattorizzata
const VectorXd& b : vettore costante b di dimension "X"
//Outputs : vettore VectorXd soluzione con fattorizzazione PALU
*/
VectorXd risolvi_conPalu( const MatrixXd& A, const VectorXd& b) {   
    
    PartialPivLU<MatrixXd> lu(A);   // fattorizzazione PALU con pivoting parziale
    return lu.solve(b);   // risolve il sistema A*x = b 
}

/*
//Funzione: risolvi_conQr
//Inputs:
const MatrixXd& A
const VectorXd& b
//Outputs : vettore VectoXd soluzione con fattorizzazione QR
*/
VectorXd risolvi_conQr( const MatrixXd& A, const VectorXd& b) {   

    HouseholderQR<MatrixXd> qr(A);   // fattorizzazione QR 
    return qr.solve(b);   // risolve il sistema A*x = b
}

/*
// Funzione: errore_relativo
// Inputs:
const VectorXd& x_esatta : primo vettore (risultato esatto)
const VectorXd& x_approx : secondo vettore (risultato approssimato)
// Outputs : errore relativo in double
*/
double errore_relativo(const VectorXd& x_esatta , const VectorXd& x_approx){   

    VectorXd differenza = x_esatta - x_approx;
    return differenza.norm()/x_esatta.norm();
}

/// MAIN

int main()
{
    // x_esatta
    VectorXd x_esatta(2);   //vettore colonna 2x1
    x_esatta << -1.0e+0, -1.0e+00;

    // Sistema 1
    MatrixXd A1(2,2);   
    A1 << 5.547001962252291e-01, -3.770900990025203e-02,
         8.320502943378437e-01, -9.992887623566787e-01;

    VectorXd b1(2);   
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;

    Vector2d x_qr1 = risolvi_conQr(A1,b1);   
    Vector2d x_palu1 = risolvi_conPalu(A1,b1);   

    double err_qr1 = errore_relativo(x_esatta, x_qr1);   // errore relativo con qr
    double err_palu1= errore_relativo(x_esatta, x_palu1);   // errore relativo con palu

    cout << "I valori si riferiscono agli errori relativi rispettivamente con fattorizzazione QR e PALU" << endl;
    cout << "Sistema_1_QR: " << err_qr1 << endl;
    cout << "Sistema_1_PALU: " << err_palu1 << endl;


    // Sistema 2
    MatrixXd A2(2,2);   
    A2 << 5.547001962252291e-01, -5.540607316466765e-01,
         8.320502943378437e-01, -8.324762492991313e-01;

    VectorXd b2(2);   
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;

    Vector2d x_qr2 = risolvi_conQr(A2,b2);   
    Vector2d x_palu2 = risolvi_conPalu(A2,b2);  

    double err_qr2 = errore_relativo(x_esatta, x_qr2);   // errore relativo con qr
    double err_palu2= errore_relativo(x_esatta, x_palu2);   // errore relativo con palu

    cout << "Sistema_2_QR: " << err_qr2 << endl;
    cout << "Sistema_2_PALU: " << err_palu2 << endl;


    // Sistema 3
    MatrixXd A3(2,2);   
    A3 << 5.547001962252291e-01, -5.547001955851905e-01,
         8.320502943378437e-01, -8.320502947645361e-01;

    VectorXd b3(2);   
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;

    Vector2d x_qr3 = risolvi_conQr(A3,b3);   
    Vector2d x_palu3 = risolvi_conPalu(A3,b3);   

    double err_qr3 = errore_relativo(x_esatta, x_qr3);   // errore relativo con qr
    double err_palu3= errore_relativo(x_esatta, x_palu3);   // errore relativo con palu

    cout << "Sistema_3_QR: " << err_qr3 << endl;
    cout << "Sistema_3_PALU: " << err_palu3 << endl;

    return 0;
}
