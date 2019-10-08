#include <cmath>
#include <iostream>
#include <vector>

int main()
{
    size_t n = 50000000;
    std::vector<double> a(n);
    std::vector<double> b(n);
    for (int i = 0; i < n; i++) {
        a[i] = sin(i) * sin(i);
        b[i] = cos(i) * cos(i);
    }

    std::vector<double> c(n);
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }

    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += c[i];
    }

    std::cout << "final result " << (sum / n) << std::endl;;

    return 0;
}
