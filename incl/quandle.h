// #include "eigen.h"


class Matrix {};

template <int N, typename T>
class Quandle
{
public:
    Quandle()
    : Q{-1} // initialize all empty with a sentinal value
    {}

    ~Quandle()
    {

    }

    // Overloaded operation for quandle operation?
    // And for inverse operation?
    // inline operation

    // T operator[]()
    // {
    //     return Q[a, b];
    // }

    T at(int a, int b)
    {

        arr
        
    }

    void update();

private:
    // Matrix<T,N,N> Q;
    T arr[(N * N + 1) / ciel(log(N)) - 1]; // Allocate memory for the array
};




class State 
{
public:
    State() = default;


    // position, currently filled in matrix
    // Any way to optimize space taken up by partially filled in matrix?
    
    int x;
    int y;
    
private:

};