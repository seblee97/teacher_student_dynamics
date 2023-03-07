#include <vector>
#include <Eigen/Dense>
#include "integrals.cpp"
#include <tuple>

using Eigen::MatrixXd;

class StudentTeacherODE
{
public:
    ODEState &state;

    int teacher_hidden;
    int student_hidden;
    bool multi_head;
    float h_learning_rate;
    float w_learning_rate;
    float timestep;
    bool train_w_layer;
    bool train_h_layer;
    // bool copy_head_at_switch;
    std::vector<float> noise_stds;

    int num_switches = 0;
    int active_teacher = 0;

    int teacher_1_offset;
    int teacher_2_offset;

    StudentTeacherODE(
        ODEState &ode_state,
        int t_hidden,
        int s_hidden,
        bool multi_h,
        float h_lr,
        float w_lr,
        float tstep,
        bool train_w,
        bool train_h,
        // bool copy_h,
        std::vector<float> noises) : state(ode_state),
                                     teacher_hidden(t_hidden),
                                     student_hidden(s_hidden),
                                     multi_head(multi_h),
                                     h_learning_rate(h_lr),
                                     w_learning_rate(w_lr),
                                     timestep(tstep),
                                     train_w_layer(train_w),
                                     train_h_layer(train_h),
                                     //  copy_head_at_switch(copy_h),
                                     noise_stds(noises)
    {
        teacher_1_offset = student_hidden;
        teacher_2_offset = student_hidden + teacher_hidden;
    }

    void set_active_teacher(int teacher_index)
    {
        this->active_teacher = teacher_index;
    }

    std::tuple<float, float> step()
    {
        // std::cout << "Taking ODE Step" << std::endl;
        this->state.step_covariance_matrix();
        // std::cout << "Stepped Cov Matrix" << std::endl;

        float e1 = error_1();
        float e2 = error_2();

        // std::cout << "error 1: " << e1 << std::endl;
        // std::cout << "error 2: " << e2 << std::endl;

        MatrixXd q_delta = MatrixXd::Constant(this->state.state["Q"].rows(), this->state.state["Q"].cols(), 0.0);
        MatrixXd r_delta = MatrixXd::Constant(this->state.state["R"].rows(), this->state.state["R"].cols(), 0.0);
        MatrixXd u_delta = MatrixXd::Constant(this->state.state["U"].rows(), this->state.state["U"].cols(), 0.0);
        MatrixXd h1_delta = MatrixXd::Constant(this->state.state["h1"].rows(), this->state.state["h1"].cols(), 0.0);
        MatrixXd h2_delta = MatrixXd::Constant(this->state.state["h2"].rows(), this->state.state["h2"].cols(), 0.0);

        if (train_w_layer)
        {
            q_delta += dq_dt();
            r_delta += dr_dt();
            u_delta += du_dt();
        }
        if (train_h_layer)
        {
            h1_delta += dh1_dt();
            h2_delta += dh2_dt();
        }

        this->state.step_order_parameter("Q", q_delta);
        this->state.step_order_parameter("R", r_delta);
        this->state.step_order_parameter("U", u_delta);
        this->state.step_order_parameter("h1", h1_delta);
        this->state.step_order_parameter("h2", h2_delta);

        std::tuple<float, float> step_errors;
        step_errors = std::make_tuple(e1, e2);
        return step_errors;
    }

    float error_1()
    {
        float error = 0;
        for (int i = 0; i < student_hidden; i++)
        {
            for (int j = 0; j < student_hidden; j++)
            {
                std::vector<int> indices{i, j};
                MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                error += 0.5 * this->state.state["h1"](i) * this->state.state["h1"](j) * sigmoid_i2(cov);
            }
        }

        for (int n = 0; n < teacher_hidden; n++)
        {
            for (int m = 0; m < teacher_hidden; m++)
            {
                std::vector<int> indices{teacher_1_offset + n, teacher_1_offset + m};
                MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                error += 0.5 * this->state.state["th1"](n) * this->state.state["th1"](m) * sigmoid_i2(cov);
            }
        }
        for (int i = 0; i < student_hidden; i++)
        {
            for (int n = 0; n < teacher_hidden; n++)
            {
                std::vector<int> indices{i, teacher_1_offset + n};
                MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                error -= this->state.state["h1"](i) * this->state.state["th1"](n) * sigmoid_i2(cov);
            }
        }
        return error;
    }

    float error_2()
    {
        float error = 0;
        for (int i = 0; i < student_hidden; i++)
        {
            for (int j = 0; j < student_hidden; j++)
            {
                std::vector<int> indices{i, j};
                MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                error += 0.5 * this->state.state["h2"](i) * this->state.state["h2"](j) * sigmoid_i2(cov);
            }
        }
        for (int p = 0; p < teacher_hidden; p++)
        {
            for (int q = 0; q < teacher_hidden; q++)
            {
                std::vector<int> indices{teacher_2_offset + p, teacher_2_offset + q};
                MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                error += 0.5 * this->state.state["th2"](p) * this->state.state["th2"](q) * sigmoid_i2(cov);
            }
        }
        for (int i = 0; i < student_hidden; i++)
        {
            for (int p = 0; p < teacher_hidden; p++)
            {
                std::vector<int> indices{i, teacher_2_offset + p};
                MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                error -= this->state.state["h2"](i) * this->state.state["th2"](p) * sigmoid_i2(cov);
            }
        }
        return error;
    }

    MatrixXd dr_dt()
    {
        // std::cout << "dR/dt" << std::endl;
        MatrixXd teacher_head(teacher_hidden, 1);
        MatrixXd student_head(student_hidden, 1);
        int offset;
        if (active_teacher == 0)
        {
            teacher_head << this->state.state["th1"];
            student_head << this->state.state["h1"];
            offset = teacher_1_offset;
        }
        else
        {
            teacher_head << this->state.state["th2"];
            student_head << this->state.state["h2"];
            offset = teacher_2_offset;
        }

        MatrixXd derivative = MatrixXd::Constant(this->state.state["R"].rows(), this->state.state["R"].cols(), 0.0);

        for (int i = 0; i < derivative.rows(); i++)
        {
            for (int n = 0; n < derivative.cols(); n++)
            {
                float in_derivative = 0.0;
                for (int m = 0; m < teacher_hidden; m++)
                {
                    std::vector<int> indices{i, teacher_1_offset + n, offset + m};
                    MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                    in_derivative += teacher_head(m) * sigmoid_i3(cov);
                }
                for (int j = 0; j < student_hidden; j++)
                {
                    std::vector<int> indices{i, teacher_1_offset + n, j};
                    MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                    in_derivative -= student_head(j) * sigmoid_i3(cov);
                }
                derivative(i, n) = timestep * w_learning_rate * student_head(i) * in_derivative;
            }
        }
        return derivative;
    }
    MatrixXd du_dt()
    {
        // std::cout << "dU/dt" << std::endl;
        MatrixXd teacher_head(teacher_hidden, 1);
        MatrixXd student_head(student_hidden, 1);
        int offset;
        if (active_teacher == 0)
        {
            teacher_head << this->state.state["th1"];
            student_head << this->state.state["h1"];
            offset = teacher_1_offset;
        }
        else
        {
            teacher_head << this->state.state["th2"];
            student_head << this->state.state["h2"];
            offset = teacher_2_offset;
        }

        MatrixXd derivative = MatrixXd::Constant(this->state.state["U"].rows(), this->state.state["U"].cols(), 0.0);

        for (int i = 0; i < derivative.rows(); i++)
        {
            for (int p = 0; p < derivative.cols(); p++)
            {
                float ip_derivative = 0.0;
                for (int m = 0; m < teacher_hidden; m++)
                {
                    std::vector<int> indices{i, teacher_2_offset + p, offset + m};
                    MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                    ip_derivative += teacher_head(m) * sigmoid_i3(cov);
                }
                for (int k = 0; k < student_hidden; k++)
                {
                    std::vector<int> indices{i, teacher_2_offset, k};
                    MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                    ip_derivative -= student_head(k) * sigmoid_i3(cov);
                }
                derivative(i, p) = timestep * w_learning_rate * student_head(i) * ip_derivative;
            }
        }
        return derivative;
    }
    MatrixXd dq_dt()
    {
        // std::cout << "dQ/dt" << std::endl;
        MatrixXd teacher_head(teacher_hidden, 1);
        MatrixXd student_head(student_hidden, 1);
        int offset;
        if (active_teacher == 0)
        {
            teacher_head << this->state.state["th1"];
            student_head << this->state.state["h1"];
            offset = teacher_1_offset;
        }
        else
        {
            teacher_head << this->state.state["th2"];
            student_head << this->state.state["h2"];
            offset = teacher_2_offset;
        }

        // MatrixXd derivative(this->state.state["Q"].rows(), this->state.state["Q"].cols());
        MatrixXd derivative = MatrixXd::Constant(this->state.state["Q"].rows(), this->state.state["Q"].cols(), 0.0);

        for (int i = 0; i < student_hidden; i++)
        {
            for (int k = i; k < student_hidden; k++)
            {
                float ik_derivative = 0.0;
                float sum_1 = 0.0;
                for (int m = 0; m < teacher_hidden; m++)
                {
                    std::vector<int> ikm_indices{i, k, offset + m};
                    MatrixXd ikm_cov = this->state.generate_sub_covariance_matrix(ikm_indices);
                    sum_1 += teacher_head(m) * student_head(i) * sigmoid_i3(ikm_cov);
                    std::vector<int> kim_indices{k, i, offset + m};
                    MatrixXd kim_cov = this->state.generate_sub_covariance_matrix(kim_indices);
                    sum_1 += teacher_head(m) * student_head(k) * sigmoid_i3(kim_cov);
                }
                for (int j = 0; j < student_hidden; j++)
                {
                    std::vector<int> ikj_indices{i, k, j};
                    MatrixXd ikj_cov = this->state.generate_sub_covariance_matrix(ikj_indices);
                    sum_1 -= student_head(j) * student_head(i) * sigmoid_i3(ikj_cov);
                    std::vector<int> kij_indices{k, i, j};
                    MatrixXd kij_cov = this->state.generate_sub_covariance_matrix(kij_indices);
                    sum_1 -= student_head(j) * student_head(k) * sigmoid_i3(kij_cov);
                }
                ik_derivative += timestep * w_learning_rate * sum_1;

                float sum_2 = 0.0;
                for (int j = 0; j < student_hidden; j++)
                {
                    for (int l = 0; l < student_hidden; l++)
                    {
                        std::vector<int> indices{i, k, j, l};
                        MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                        sum_2 += student_head(j) * student_head(l) * sigmoid_i4(cov);
                    }
                }
                for (int m = 0; m < teacher_hidden; m++)
                {
                    for (int n = 0; n < teacher_hidden; n++)
                    {
                        std::vector<int> indices{i, k, offset + m, offset + n};
                        MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                        sum_2 += teacher_head(m) * teacher_head(n) * sigmoid_i4(cov);
                    }
                }
                for (int m = 0; m < teacher_hidden; m++)
                {
                    for (int j = 0; j < student_hidden; j++)
                    {
                        std::vector<int> indices{i, k, j, offset + m};
                        MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                        sum_2 -= 2 * teacher_head(m) * student_head(j) * sigmoid_i4(cov);
                    }
                }
                // noise term
                std::vector<int> indices{i, k};
                MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                sum_2 += pow(noise_stds[active_teacher], 2) * sigmoid_j2(cov);

                ik_derivative += timestep * pow(w_learning_rate, 2) * student_head(i) * student_head(k) * sum_2;
                derivative(i, k) = ik_derivative;
            }
        }

        // copy elements into opposite diagonal
        derivative.triangularView<Eigen::Lower>() = derivative.transpose();

        return derivative;
    }
    MatrixXd dh1_dt()
    {
        // std::cout << "dh1/dt" << std::endl;
        // MatrixXd derivative(this->state.state["h1"].rows(), this->state.state["h1"].cols());
        MatrixXd derivative = MatrixXd::Constant(this->state.state["h1"].rows(), this->state.state["h1"].cols(), 0.0);
        if (train_h_layer and active_teacher == 0)
        {
            for (int i = 0; i < student_hidden; i++)
            {
                float i_derivative = 0.0;
                for (int m = 0; m < teacher_hidden; m++)
                {
                    std::vector<int> indices{i, teacher_1_offset + m};
                    MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                    i_derivative += this->state.state["th1"](m) * sigmoid_i2(cov);
                }
                for (int k = 0; k < student_hidden; k++)
                {
                    std::vector<int> indices{i, k};
                    MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                    i_derivative -= this->state.state["h1"](k) * sigmoid_i2(cov);
                }
                derivative(i) = timestep * h_learning_rate * i_derivative;
            }
        }
        return derivative;
    }
    MatrixXd dh2_dt()
    {
        // std::cout << "dh2/dt" << std::endl;
        MatrixXd derivative = MatrixXd::Constant(this->state.state["h2"].rows(), this->state.state["h2"].cols(), 0.0);

        if (train_h_layer and active_teacher == 1)
        {
            for (int i = 0; i < student_hidden; i++)
            {
                float i_derivative = 0.0;
                for (int p = 0; p < teacher_hidden; p++)
                {
                    std::vector<int> indices{i, teacher_2_offset + p};
                    MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                    i_derivative += this->state.state["th2"](p) * sigmoid_i2(cov);
                }
                for (int k = 0; k < student_hidden; k++)
                {
                    std::vector<int> indices{i, k};
                    MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                    i_derivative -= this->state.state["h2"](k) * sigmoid_i2(cov);
                }
                derivative(i) = timestep * h_learning_rate * i_derivative;
            }
        }
        return derivative;
    }
};
