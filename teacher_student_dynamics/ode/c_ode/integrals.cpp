#include <Eigen/Dense>
#include <math.h>
#include <cmath>

using Eigen::MatrixXd;

float lambda_4(MatrixXd covariance)
{
    return (1 + covariance(0, 0)) * (1 + covariance(1, 1)) - pow(covariance(0, 1), 2);
}

float lambda_0(MatrixXd covariance)
{
    float term_1 = lambda_4(covariance) * covariance(2, 3);
    float term_2 = covariance(1, 2) * covariance(1, 3) * (1 + covariance(0, 0));
    float term_3 = covariance(0, 2) * covariance(0, 3) * (1 + covariance(1, 1));
    float term_4 = covariance(0, 1) * covariance(0, 2) * covariance(1, 3);
    float term_5 = covariance(0, 1) * covariance(0, 3) * covariance(1, 2);
    return term_1 - term_2 - term_3 + term_4 + term_5;
}

float lambda_1(MatrixXd covariance)
{
    float term_1 = lambda_4(covariance) * (1 + covariance(2, 2));
    float term_2 = pow(covariance(1, 2), 2) * (1 + covariance(0, 0));
    float term_3 = pow(covariance(0, 2), 2) * (1 + covariance(1, 1));
    float term_4 = 2 * covariance(0, 1) * covariance(0, 2) * covariance(1, 2);
    return term_1 - term_2 - term_3 + term_4;
}

float lambda_2(MatrixXd covariance)
{
    float term_1 = lambda_4(covariance) * (1 + covariance(3, 3));
    float term_2 = pow(covariance(1, 3), 2) * (1 + covariance(0, 0));
    float term_3 = pow(covariance(0, 3), 2) * (1 + covariance(1, 1));
    float term_4 = 2 * covariance(0, 1) * covariance(0, 3) * covariance(1, 3);
    return term_1 - term_2 - term_3 + term_4;
}

float lambda_3(MatrixXd covariance)
{
    return (1 + covariance(0, 0)) * (1 + covariance(2, 2)) - pow(covariance(0, 2), 2);
}

float sigmoid_i3(MatrixXd covariance)
{
    float nom = 2 * (covariance(1, 2) * (1 + covariance(0, 0)) - covariance(0, 1) * covariance(0, 2));
    float den = M_PI * sqrt(lambda_3(covariance)) * (1 + covariance(0, 0));
    return nom / den;
}

float sigmoid_i4(MatrixXd covariance)
{
    float nom = 4 * asin(lambda_0(covariance) / sqrt(lambda_1(covariance) * lambda_2(covariance)));
    float den = pow(M_PI, 2) * sqrt(lambda_4(covariance));
    return nom / den;
}

float sigmoid_i2(MatrixXd covariance)
{
    float nom = covariance(0, 1);
    float den = sqrt(1 + covariance(0, 0)) * sqrt(1 + covariance(1, 1));
    return 2 * asin(nom / den) / M_PI;
}

float sigmoid_j2(MatrixXd covariance)
{
    float base_term_1 = 1 + covariance(0, 0) + covariance(1, 1);
    float base_term_2 = covariance(0, 0) * covariance(1, 1) - pow(covariance(0, 1), 2);
    return 2 * pow(base_term_1 + base_term_2, -0.5) / M_PI;
}