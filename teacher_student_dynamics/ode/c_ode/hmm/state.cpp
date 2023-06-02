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

class HMMODEState
{
private:
    int teacher_hidden;
    int student_hidden;
    bool multi_head;
    int num_bins;

    Matrix<double, Dynamic, Dynamic> W;
    Matrix<double, Dynamic, Dynamic> Sigma1;
    Matrix<double, Dynamic, Dynamic> Sigma2;
    // Matrix<double, Dynamic, Dynamic> Omega1;
    // Matrix<double, Dynamic, Dynamic> Omega2;
    // Matrix<double, Dynamic, Dynamic> S1;
    // Matrix<double, Dynamic, Dynamic> S2;
    // Matrix<double, Dynamic, Dynamic> r_density;
    // Matrix<double, Dynamic, Dynamic> u_density;
    Matrix<double, Dynamic, Dynamic> Q;
    Matrix<double, Dynamic, Dynamic> R;
    Matrix<double, Dynamic, Dynamic> U;
    Matrix<double, Dynamic, Dynamic> T;
    Matrix<double, Dynamic, Dynamic> H;
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

    HMMODEState(int t_hidden, int s_hidden, bool multi_h, int n_bins, std::string order_parameter_paths) : teacher_hidden(t_hidden), student_hidden(s_hidden), multi_head(multi_h), num_bins(n_bins)
    {
        resize_matrices();
        populate_state_map();
        read_state_from_file(order_parameter_paths);

        step_covariance_matrix();
    }

    HMMODEState() = default;

    void resize_matrices()
    {
        // this->u_density.resize(num_bins, student_hidden * teacher_hidden);
        // this->r_density.resize(num_bins, student_hidden * teacher_hidden);
        this->W.resize(student_hidden, student_hidden);
        this->Sigma1.resize(student_hidden, student_hidden);
        this->Sigma2.resize(student_hidden, student_hidden);
        // this->S1.resize(student_hidden, latent_dim);
        // this->S2.resize(student_hidden, latent_dim);
        this->Q.resize(student_hidden, student_hidden);
        this->R.resize(student_hidden, teacher_hidden);
        this->U.resize(student_hidden, teacher_hidden);
        this->T.resize(teacher_hidden, teacher_hidden);
        this->H.resize(teacher_hidden, teacher_hidden);
        this->V.resize(teacher_hidden, teacher_hidden);
        this->h1.resize(student_hidden, 1);
        if (multi_head)
        {
            this->h2.resize(student_hidden, 1);
        }
        this->th1.resize(teacher_hidden, 1);
        this->th2.resize(teacher_hidden, 1);

        this->block_1.resize(student_hidden, student_hidden + 2 * teacher_hidden);
        this->block_2.resize(teacher_hidden, student_hidden + 2 * teacher_hidden);
        this->block_3.resize(teacher_hidden, student_hidden + 2 * teacher_hidden);

        this->covariance_matrix.resize(student_hidden + 2 * teacher_hidden, student_hidden + 2 * teacher_hidden);

        std::cout << "covariance matrix successfully resized." << std::endl;
    }

    void populate_state_map()
    {
        // this->state.insert({"r_density", this->r_density});
        // this->state.insert({"u_density", this->u_density});
        this->state.insert({"W", this->W});
        this->state.insert({"Sigma1", this->Sigma1});
        this->state.insert({"Sigma2", this->Sigma2});
        this->state.insert({"Q", this->Q});
        this->state.insert({"R", this->R});
        this->state.insert({"U", this->U});
        this->state.insert({"T", this->T});
        this->state.insert({"H", this->H});
        this->state.insert({"V", this->V});
        this->state.insert({"h1", this->h1});
        if (multi_head)
        {
            this->state.insert({"h2", this->h2});
        }
        this->state.insert({"th1", this->th1});
        this->state.insert({"th2", this->th2});

        std::cout << "state map populated." << std::endl;
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

    void step_covariance_matrix()
    {
        // std::cout << "Stepping Covariance Matrix" << std::endl;
        this->block_1 << this->state["Q"], this->state["R"], this->state["U"];
        this->block_2 << this->state["R"].transpose(), this->state["T"], this->state["V"];
        this->block_3 << this->state["U"].transpose(), this->state["V"].transpose(), this->state["H"];
        // std::cout << "Stepped Covariance Matrix" << std::endl;

        this->covariance_matrix << this->block_1, this->block_2, this->block_3;
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

        std::cout << "state read from files." << std::endl;
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
        std::cout << "order parameter " << order_parameter_name << " read from file." << std::endl;
    }
};
