#include <iostream>
#include <opencv2/opencv.hpp>

//Protoypes
void MainFromCppExample ();

using namespace std;

int main()
{
    cout << "\n\nC++ CV Test 01: Program Started.\n\n";
    
    MainFromCppExample ();
    
    system("pause");
    cout << "\n\nC++ CV Test 01: Program Ended.\n\n";
    return 0;
}

void MainFromCppExample ()
{
    cout << "\n\nC++ CV Test 01: Running Main() code from C++ Example.";
}
