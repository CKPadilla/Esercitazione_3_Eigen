#include <iostream>
#include "Eigen/Eigen"

using namespace std;
using namespace Eigen;

/* 
La consegna mi dà 3 matrici A e 3 matrici b che hanno relativamente dimensione costante.
Provo a costruire una funzione che calcoli la soluzione di sistemi di qualsiasi dimensione 
attraverso PALU e QR come fattorizzazione.
*/

//funzione per PALU

VectorXd sistemaPALU( const MatrixXd& A, const VectorXd& b) {   //prendo come parametri la matrice A e il vettore col costante b (entrambi in const poichè non vanno modificati)
    
    PartialPivLU<MatrixXd> lu(A);   // memorizzo nell'oggetto lu il risultato della fattorizzazione con pivoting parziale di A
    return lu.solve(b);   // viene risolto il sistema con A fattorizzata 
}

// funzione per QR (struttura simile a PALU)

VectorXd sistemaQR( const MatrixXd& A, const VectorXd& b) {   //prendo come parametri la matrice A e il vettore col costante b (entrambi in const poichè non vanno modificati)

    HouseholderQR<MatrixXd> qr(A);   // memorizzo nell'oggetto qr il risultato della fattorizzazione di A
    return qr.solve(b);   // viene risolto il sistema con A fattorizzata 
}

// Funzione che calcola l'errore relativo
double errore(const VectorXd& x0 , const VectorXd& x1){   // prende come parametri x0 = risultato esatto e x1= risultato approx

    VectorXd diff = x0 - x1;
    return diff.norm()/x0.norm();

}

int main()
{
    // scrivo la soluzione esatta
    VectorXd x_ex;
    x_ex << -1.0e+0, -1.0e+00;

    // primo sistema
    MatrixXd A1(2,2);   // creo la matrice A
    A1 << 5.547001962252291e-01, -3.770900990025203e-02,
         8.320502943378437e-01, -9.992887623566787e-01;

    VectorXd b1(2);   // creo il vettore coeff b
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;

    VectorXd x_qr1 = sistemaQR(A1,b1);   // risolvo con fattorizzazione qr
    VectorXd x_palu1 = sistemaPALU(A1,b1);   // risolvo con fattorizzazione palu

    double err_qr1 = errore(x_ex, x_qr1);   // errore relativo con qr
    double err_palu1= errore(x_ex, x_palu1);   // errore relativo con palu

    cout << "I valori si riferiscono agli errori relativi rispettivamente con fattorizzazione QR e PALU" << endl;
    cout << "Sistema_1_QR: " << err_qr1 << endl;
    cout << "Sistema_1_PALU: " << err_palu1 << endl;


    // secondo sistema
    MatrixXd A2(2,2);   // creo la matrice A
    A2 << 5.547001962252291e-01, -5.540607316466765e-01,
         8.320502943378437e-01, -8.324762492991313e-01;

    VectorXd b2(2);   // creo il vettore coeff b
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;

    VectorXd x_qr2 = sistemaQR(A2,b2);   // risolvo con fattorizzazione qr
    VectorXd x_palu2 = sistemaPALU(A2,b2);   // risolvo con fattorizzazione palu

    double err_qr2 = errore(x_ex, x_qr2);   // errore relativo con qr
    double err_palu2= errore(x_ex, x_palu2);   // errore relativo con palu

    cout << "Sistema_2_QR: " << err_qr2 << endl;
    cout << "Sistema_2_PALU: " << err_palu2 << endl;


    // terzo sistema

    MatrixXd A3(2,2);   // creo la matrice A
    A3 << 5.547001962252291e-01, -5.547001955851905e-01,
         8.320502943378437e-01, -8.320502947645361e-01;

    VectorXd b3(2);   // creo il vettore coeff b
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;

    VectorXd x_qr3 = sistemaQR(A3,b3);   // risolvo con fattorizzazione qr
    VectorXd x_palu3 = sistemaPALU(A3,b3);   // risolvo con fattorizzazione palu

    double err_qr3 = errore(x_ex, x_qr3);   // errore relativo con qr
    double err_palu3= errore(x_ex, x_palu3);   // errore relativo con palu

    cout << "Sistema_3_QR: " << err_qr3 << endl;
    cout << "Sistema_3_PALU: " << err_palu3 << endl;

    return 0;
}
