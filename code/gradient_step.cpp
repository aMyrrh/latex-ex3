#include <iostream>
#include <vector>

struct Model {
    double w = 0.0;
    double b = 0.0;
};

void gradient_step(Model& model, const std::vector<double>& x,
                   const std::vector<double>& y, double learning_rate) {
    double grad_w = 0.0;
    double grad_b = 0.0;
    const double n = static_cast<double>(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        double error = model.w * x[i] + model.b - y[i];
        grad_w += 2.0 * error * x[i] / n;
        grad_b += 2.0 * error / n;
    }
    model.w -= learning_rate * grad_w;
    model.b -= learning_rate * grad_b;
}

int main() {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {3.6, 6.1, 8.4, 11.1, 13.4};
    Model model;
    for (int epoch = 0; epoch < 2000; ++epoch) {
        gradient_step(model, x, y, 0.02);
    }
    std::cout << "w = " << model.w << ", b = " << model.b << std::endl;
    return 0;
}
