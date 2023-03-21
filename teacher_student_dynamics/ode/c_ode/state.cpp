#include <vector>
#include <sstream>
#include <list>
#include <iterator>
#include <map>
#include <string>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>

using Eigen::Dynamic;
using Eigen::Matrix;
using Eigen::MatrixXd;

class ODEState
{
private:
    int teacher_hidden;
    int student_hidden;
    bool multi_head;

    Matrix<double, Dynamic, Dynamic> Q;
    Matrix<double, Dynamic, Dynamic> R;
    Matrix<double, Dynamic, Dynamic> U;
    Matrix<double, Dynamic, Dynamic> S;
    Matrix<double, Dynamic, Dynamic> T;
    Matrix<double, Dynamic, Dynamic> V;
    Matrix<double, Dynamic, Dynamic> h1;
    Matrix<double, Dynamic, Dynamic> h2;
    Matrix<double, Dynamic, Dynamic> th1;
    Matrix<double, Dynamic, Dynamic> th2;

    Matrix<double, Dynamic, Dynamic> block_1;
    Matrix<double, Dynamic, Dynamic> block_2;
    Matrix<double, Dynamic, Dynamic> block_3;
    Matrix<double, Dynamic, Dynamic> block_4;
    Matrix<double, Dynamic, Dynamic> block_5;
    Matrix<double, Dynamic, Dynamic> block_6;

    Matrix<double, Dynamic, Dynamic> covariance_matrix;

public:
    // initialise state, map between order parameter name and values
    std::map<std::string, Matrix<double, Dynamic, Dynamic>> state;

    ODEState(int t_hidden, int s_hidden, bool multi_h, std::string order_parameter_paths) : teacher_hidden(t_hidden), student_hidden(s_hidden), multi_head(multi_h)
    {
        resize_matrices();
        populate_state_map();
        read_state_from_file(order_parameter_paths);

        step_covariance_matrix();
    }

    ODEState() = default;

    void resize_matrices()
    {
        this->Q.resize(student_hidden, student_hidden);
        this->R.resize(student_hidden, teacher_hidden);
        this->S.resize(teacher_hidden, teacher_hidden);
        this->U.resize(student_hidden, teacher_hidden);
        this->T.resize(teacher_hidden, teacher_hidden);
        this->V.resize(teacher_hidden, teacher_hidden);
        this->h1.resize(student_hidden, 1);
        if (multi_head)
        {
            this->h2.resize(student_hidden, 1);
        }
        this->th1.resize(teacher_hidden, 1);
        this->th2.resize(teacher_hidden, 1);

        this->block_1.resize(student_hidden, 2 * (student_hidden + 2 * teacher_hidden));
        this->block_2.resize(teacher_hidden, 2 * (student_hidden + 2 * teacher_hidden));
        this->block_3.resize(teacher_hidden, 2 * (student_hidden + 2 * teacher_hidden));
        this->block_4.resize(student_hidden, 2 * (student_hidden + 2 * teacher_hidden));
        this->block_5.resize(teacher_hidden, 2 * (student_hidden + 2 * teacher_hidden));
        this->block_6.resize(teacher_hidden, 2 * (student_hidden + 2 * teacher_hidden));

        this->covariance_matrix.resize(2 * (student_hidden + 2 * teacher_hidden), 2 * (student_hidden + 2 * teacher_hidden));
    }

    void populate_state_map()
    {
        this->state.insert({"Q", this->Q});
        this->state.insert({"R", this->R});
        this->state.insert({"U", this->U});
        this->state.insert({"S", this->S});
        this->state.insert({"T", this->T});
        this->state.insert({"V", this->V});
        this->state.insert({"h1", this->h1});
        if (multi_head)
        {
            this->state.insert({"h2", this->h2});
        }
        this->state.insert({"th1", this->th1});
        this->state.insert({"th2", this->th2});
    }

    MatrixXd getCovarianceMatrix()
    {
        return covariance_matrix;
    }

    Matrix<double, Dynamic, Dynamic> generate_sub_covariance_matrix(std::vector<int> indices)
    {
        MatrixXd matrix = MatrixXd::Constant(indices.size(), indices.size(), 0);
        for (int i = 0; i < indices.size(); ++i)
        {
            for (int j = 0; j < indices.size(); ++j)
            {
                // std::cout << i << j << indices[i] << indices[j] << std::endl;
                matrix(i, j) = covariance_matrix(indices[i], indices[j]);
            }
        }
        return matrix;
    }

    void step_covariance_matrix(float input_noise = 0.0)
    {
        // std::cout << "Stepping Covariance Matrix" << std::endl;
        this->block_1 << this->state["Q"], this->state["R"], this->state["U"], this->state["Q"], this->state["R"], this->state["U"];
        this->block_2 << this->state["R"].transpose(), this->state["T"], this->state["V"], this->state["R"].transpose(), this->state["T"], this->state["V"];
        this->block_3 << this->state["U"].transpose(), this->state["V"].transpose(), this->state["S"], this->state["U"].transpose(), this->state["V"].transpose(), this->state["S"];
        this->block_4 << this->state["Q"], this->state["R"], this->state["U"], (1 + pow(input_noise, 2)) * this->state["Q"], (1 + pow(input_noise, 2)) * this->state["R"], (1 + pow(input_noise, 2)) * this->state["U"];
        this->block_5 << this->state["R"].transpose(), this->state["T"], this->state["V"].transpose(), (1 + pow(input_noise, 2)) * this->state["R"].transpose(), (1 + pow(input_noise, 2)) * this->state["T"], (1 + pow(input_noise, 2)) * this->state["V"].transpose();
        this->block_6 << this->state["U"].transpose(), this->state["V"], this->state["S"], (1 + pow(input_noise, 2)) * this->state["U"].transpose(), (1 + pow(input_noise, 2)) * this->state["V"], (1 + pow(input_noise, 2)) * this->state["S"];
        // std::cout << "Stepped Covariance Matrix" << std::endl;

        this->covariance_matrix << this->block_1, this->block_2, this->block_3, this->block_4, this->block_5, this->block_6;
    }

    void step_order_parameter(std::string order_parameter, MatrixXd delta)
    {
        this->state[order_parameter] = this->state[order_parameter] + delta;
    }

private:
    void read_state_from_file(std::string order_parameter_paths)
    {
        std::string line;
        std::ifstream file(order_parameter_paths);

        if (file.is_open())
        {
            while (std::getline(file, line))
            {
                std::vector<std::string> tokens;
                std::string token;
                std::istringstream tokenStream(line);
                while (std::getline(tokenStream, token, ','))
                {
                    tokens.push_back(token);
                }
                read_order_parameter_from_file(tokens[1], tokens[0]);
            }
            file.close();
        }
    }

    void read_order_parameter_from_file(std::string path, std::string order_parameter_name)
    {
        std::string line;
        std::ifstream file(path);

        int row_index = 0;
        int column_index = 0;

        if (file.is_open())
        {
            while (std::getline(file, line))
            {
                column_index = 0;
                std::string token;
                std::istringstream tokenStream(line);
                while (std::getline(tokenStream, token, ','))
                {
                    this->state[order_parameter_name](row_index, column_index) = std::stof(token);
                    column_index += 1;
                }
                row_index += 1;
            }
            file.close();
        }
        else
        {
            std::cout << "Unable to open file" << std::endl;
        }
    }
};
