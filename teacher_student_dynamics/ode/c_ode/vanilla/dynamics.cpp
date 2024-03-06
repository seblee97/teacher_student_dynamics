#include <vector>
#include <Eigen/Dense>
#include "../integrals.cpp"
#include <tuple>
// #include <omp.h>

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
    std::vector<float> input_noise_stds;
    std::vector<float> noise_stds;
    std::vector<int> freeze_units;

    int active_teacher;

    int teacher_1_offset;
    int teacher_2_offset;
    int input_noise_offset;

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
        std::vector<float> input_noises,
        std::vector<float> noises,
        std::vector<int> freeze) : state(ode_state),
                                   teacher_hidden(t_hidden),
                                   student_hidden(s_hidden),
                                   multi_head(multi_h),
                                   h_learning_rate(h_lr),
                                   w_learning_rate(w_lr),
                                   timestep(tstep),
                                   train_w_layer(train_w),
                                   train_h_layer(train_h),
                                   //  copy_head_at_switch(copy_h),
                                   input_noise_stds(input_noises),
                                   noise_stds(noises),
                                   freeze_units(freeze)
    {
        teacher_1_offset = student_hidden;
        teacher_2_offset = student_hidden + teacher_hidden;
        input_noise_offset = student_hidden + 2 * teacher_hidden;
        set_active_teacher(0);
    }

    void set_active_teacher(int teacher_index)
    {
        this->active_teacher = teacher_index;
        std::cout << "Teacher Index: " << active_teacher << std::endl;
        std::cerr << "Teacher Index: " << active_teacher << std::endl;
        std::cout << "Input Noise: " << input_noise_stds[active_teacher] << std::endl;
        std::cout << "Output Noise: " << noise_stds[active_teacher] << std::endl;
        std::cerr << "Input Noise: " << input_noise_stds[active_teacher] << std::endl;
        std::cerr << "Output Noise: " << noise_stds[active_teacher] << std::endl;
    }

    int get_active_teacher()
    {
        return this->active_teacher;
    }

    std::tuple<float, float> step()
    {
        // std::cout << "Taking ODE Step" << std::endl;
        this->state.step_covariance_matrix(input_noise_stds[active_teacher]);
        // std::cout << "Stepped Cov Matrix" << std::endl;

        float e1 = 0;
        float e2 = 0;

        e1 += error_1();
        e2 += error_2();

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
            if (multi_head)
            {
                h2_delta += dh2_dt();
            }
        }

        this->state.step_order_parameter("Q", q_delta);
        this->state.step_order_parameter("R", r_delta);
        this->state.step_order_parameter("U", u_delta);
        this->state.step_order_parameter("h1", h1_delta);

        if (multi_head)
        {
            this->state.step_order_parameter("h2", h2_delta);
        }

        std::tuple<float, float> step_errors;
        step_errors = std::make_tuple(e1, e2);
        return step_errors;
    }

    float error_1()
    {
        float error = 0;
#pragma omp parallel sections reduction(+ : error)
        {
#pragma omp section
#pragma omp parallel for collapse(2)
            for (int i = 0; i < student_hidden; i++)
            {
                for (int j = 0; j < student_hidden; j++)
                {
                    std::vector<int> indices{i, j};
                    MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                    error += 0.5 * this->state.state["h1"](i) * this->state.state["h1"](j) * sigmoid_i2(cov);
                }
            }

#pragma omp section
#pragma omp parallel for collapse(2)
            for (int n = 0; n < teacher_hidden; n++)
            {
                for (int m = 0; m < teacher_hidden; m++)
                {
                    std::vector<int> indices{teacher_1_offset + n, teacher_1_offset + m};
                    MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                    error += 0.5 * this->state.state["th1"](n) * this->state.state["th1"](m) * sigmoid_i2(cov);
                }
            }
#pragma omp section
#pragma omp parallel for collapse(2)
            for (int i = 0; i < student_hidden; i++)
            {
                for (int n = 0; n < teacher_hidden; n++)
                {
                    std::vector<int> indices{i, teacher_1_offset + n};
                    MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                    error -= this->state.state["h1"](i) * this->state.state["th1"](n) * sigmoid_i2(cov);
                }
            }
        }
        return error;
    }

    float error_2()
    {
        Matrix<double, Dynamic, Dynamic> head;
        head.resize(student_hidden, 1);
        if (multi_head)
        {
            head = this->state.state["h2"];
        }
        else
        {
            head = this->state.state["h1"];
        }

        float error = 0;
#pragma omp parallel sections reduction(+ : error)
        {
#pragma omp section
#pragma omp parallel for collapse(2)
            for (int i = 0; i < student_hidden; i++)
            {
                for (int j = 0; j < student_hidden; j++)
                {
                    std::vector<int> indices{i, j};
                    MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                    error += 0.5 * head(i) * head(j) * sigmoid_i2(cov);
                }
            }
#pragma omp section
#pragma omp parallel for collapse(2)
            for (int p = 0; p < teacher_hidden; p++)
            {
                for (int q = 0; q < teacher_hidden; q++)
                {
                    std::vector<int> indices{teacher_2_offset + p, teacher_2_offset + q};
                    MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                    error += 0.5 * this->state.state["th2"](p) * this->state.state["th2"](q) * sigmoid_i2(cov);
                }
            }
#pragma omp section
#pragma omp parallel for collapse(2)
            for (int i = 0; i < student_hidden; i++)
            {
                for (int p = 0; p < teacher_hidden; p++)
                {
                    std::vector<int> indices{i, teacher_2_offset + p};
                    MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                    error -= head(i) * this->state.state["th2"](p) * sigmoid_i2(cov);
                }
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
            offset = teacher_2_offset;
            if (multi_head)
            {
                student_head << this->state.state["h2"];
            }
            else
            {
                student_head << this->state.state["h1"];
            }
        }

        MatrixXd derivative = MatrixXd::Constant(this->state.state["R"].rows(), this->state.state["R"].cols(), 0.0);

#pragma omp parallel for collapse(2)
        for (int i = std::max(0, freeze_units[active_teacher]); i < derivative.rows(); i++)
        {
            for (int n = 0; n < derivative.cols(); n++)
            {
                float in_derivative = 0.0;
#pragma omp parallel sections reduction(+ : in_derivative)
                {
#pragma omp section
                    for (int m = 0; m < teacher_hidden; m++)
                    {
                        std::vector<int> indices{i + input_noise_offset, teacher_1_offset + n + input_noise_offset, offset + m};
                        MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                        in_derivative += teacher_head(m) * sigmoid_i3(cov);
                    }
#pragma omp section
                    for (int j = 0; j < student_hidden; j++)
                    {
                        std::vector<int> indices{i + input_noise_offset, teacher_1_offset + n + input_noise_offset, j + input_noise_offset};
                        MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                        in_derivative += -(student_head(j) * sigmoid_i3(cov));
                    }
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
            offset = teacher_2_offset;
            if (multi_head)
            {
                student_head << this->state.state["h2"];
            }
            else
            {
                student_head << this->state.state["h1"];
            }
        }

        MatrixXd derivative = MatrixXd::Constant(this->state.state["U"].rows(), this->state.state["U"].cols(), 0.0);

#pragma omp parallel for collapse(2)
        for (int i = std::max(0, freeze_units[active_teacher]); i < derivative.rows(); i++)
        {
            for (int p = 0; p < derivative.cols(); p++)
            {
                float ip_derivative = 0.0;
#pragma omp parallel sections reduction(+ : ip_derivative)
                {
#pragma omp section
                    for (int m = 0; m < teacher_hidden; m++)
                    {
                        std::vector<int> indices{i + input_noise_offset, teacher_2_offset + p + input_noise_offset, offset + m};
                        MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                        ip_derivative += teacher_head(m) * sigmoid_i3(cov);
                    }
#pragma omp section
                    for (int k = 0; k < student_hidden; k++)
                    {
                        std::vector<int> indices{i + input_noise_offset, teacher_2_offset + p + input_noise_offset, k + input_noise_offset};
                        MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                        ip_derivative += -student_head(k) * sigmoid_i3(cov);
                    }
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
            offset = teacher_2_offset;
            if (multi_head)
            {
                student_head << this->state.state["h2"];
            }
            else
            {
                student_head << this->state.state["h1"];
            }
        }

        // MatrixXd derivative(this->state.state["Q"].rows(), this->state.state["Q"].cols());
        MatrixXd derivative = MatrixXd::Constant(this->state.state["Q"].rows(), this->state.state["Q"].cols(), 0.0);

#pragma omp parallel for
        for (int i = 0; i < student_hidden; i++)
        {
            for (int k = i; k < student_hidden; k++)
            {
                float ik_derivative = 0.0;
                float sum_1_factor = timestep * w_learning_rate;
                float sum_2_factor = timestep * pow(w_learning_rate, 2) * (1 + pow(input_noise_stds[active_teacher], 2)) * student_head(i) * student_head(k);
#pragma omp parallel sections reduction(+ : ik_derivative)
                {
#pragma omp section
                    for (int m = 0; m < teacher_hidden; m++)
                    {
                        // if i not a frozen unit
                        if (i >= freeze_units[active_teacher])
                        {
                            std::vector<int> ikm_indices{i + input_noise_offset, k + input_noise_offset, offset + m};
                            MatrixXd ikm_cov = this->state.generate_sub_covariance_matrix(ikm_indices);
                            ik_derivative += sum_1_factor * teacher_head(m) * student_head(i) * sigmoid_i3(ikm_cov);
                        }
                        // if k not a frozen unit
                        if (k >= freeze_units[active_teacher])
                        {
                            std::vector<int> kim_indices{k + input_noise_offset, i + input_noise_offset, offset + m};
                            MatrixXd kim_cov = this->state.generate_sub_covariance_matrix(kim_indices);
                            ik_derivative += sum_1_factor * teacher_head(m) * student_head(k) * sigmoid_i3(kim_cov);
                        }
                    }
#pragma omp section
                    for (int j = 0; j < student_hidden; j++)
                    {
                        // if i not a frozen unit
                        if (i >= freeze_units[active_teacher])
                        {
                            std::vector<int> ikj_indices{i + input_noise_offset, k + input_noise_offset, j + input_noise_offset};
                            MatrixXd ikj_cov = this->state.generate_sub_covariance_matrix(ikj_indices);
                            ik_derivative += -sum_1_factor * student_head(j) * student_head(i) * sigmoid_i3(ikj_cov);
                        }
                        // if k not a frozen unit
                        if (k >= freeze_units[active_teacher])
                        {
                            std::vector<int> kij_indices{k + input_noise_offset, i + input_noise_offset, j + input_noise_offset};
                            MatrixXd kij_cov = this->state.generate_sub_covariance_matrix(kij_indices);
                            ik_derivative += -sum_1_factor * student_head(j) * student_head(k) * sigmoid_i3(kij_cov);
                        }
                    }

#pragma omp section
                    // if i and k both unfrozen
                    if (i >= freeze_units[active_teacher] and k >= freeze_units[active_teacher])
                    {
#pragma omp parallel for collapse(2)
                        for (int j = 0; j < student_hidden; j++)
                        {
                            for (int l = 0; l < student_hidden; l++)
                            {
                                std::vector<int> indices{i + input_noise_offset, k + input_noise_offset, j + input_noise_offset, l + input_noise_offset};
                                MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                                ik_derivative += sum_2_factor * student_head(j) * student_head(l) * sigmoid_i4(cov);
                            }
                        }
                    }
#pragma omp section
                    // if i and k both unfrozen
                    if (i >= freeze_units[active_teacher] and k >= freeze_units[active_teacher])
                    {
#pragma omp parallel for collapse(2)
                        for (int m = 0; m < teacher_hidden; m++)
                        {
                            for (int n = 0; n < teacher_hidden; n++)
                            {
                                std::vector<int> indices{i + input_noise_offset, k + input_noise_offset, offset + m, offset + n};
                                MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                                ik_derivative += sum_2_factor * teacher_head(m) * teacher_head(n) * sigmoid_i4(cov);
                            }
                        }
                    }
#pragma omp section
                    // if i and k both unfrozen
                    if (i >= freeze_units[active_teacher] and k >= freeze_units[active_teacher])
                    {
#pragma omp parallel for collapse(2)
                        for (int m = 0; m < teacher_hidden; m++)
                        {
                            for (int j = 0; j < student_hidden; j++)
                            {
                                std::vector<int> indices{i + input_noise_offset, k + input_noise_offset, j + input_noise_offset, offset + m};
                                MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                                ik_derivative += -sum_2_factor * 2 * teacher_head(m) * student_head(j) * sigmoid_i4(cov);
                            }
                        }
                    }
#pragma omp section
                    // if i and k both unfrozen
                    if (i >= freeze_units[active_teacher] and k >= freeze_units[active_teacher])
                    {
                        // noise term
                        std::vector<int> indices{i + input_noise_offset, k + input_noise_offset};
                        MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                        ik_derivative += sum_2_factor * pow(noise_stds[active_teacher], 2) * sigmoid_j2(cov);
                    }
                }
                derivative(i, k) = ik_derivative;
            }
        }

        // copy elements into opposite diagonal
        derivative.triangularView<Eigen::Lower>() = derivative.transpose();

        return derivative;
    }
    MatrixXd dh1_dt()
    {
        MatrixXd teacher_head(teacher_hidden, 1);
        int offset;
        if (active_teacher == 0)
        {
            teacher_head << this->state.state["th1"];
            offset = teacher_1_offset;
        }
        else
        {
            teacher_head << this->state.state["th2"];
            offset = teacher_2_offset;
        }
        // std::cout << "dh1/dt" << std::endl;
        // MatrixXd derivative(this->state.state["h1"].rows(), this->state.state["h1"].cols());
        MatrixXd derivative = MatrixXd::Constant(this->state.state["h1"].rows(), this->state.state["h1"].cols(), 0.0);
        if (train_h_layer and (active_teacher == 0) or (not multi_head))
        {
#pragma omp parallel for
            for (int i = 0; i < student_hidden; i++)
            {
                float i_derivative = 0.0;
#pragma omp parallel sections reduction(+ : i_derivative)
                {
#pragma omp section
                    for (int m = 0; m < teacher_hidden; m++)
                    {
                        std::vector<int> indices{i + input_noise_offset, teacher_1_offset + m};
                        MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                        i_derivative += teacher_head(m) * sigmoid_i2(cov);
                    }
#pragma omp section
                    for (int k = 0; k < student_hidden; k++)
                    {
                        std::vector<int> indices{i + input_noise_offset, k + input_noise_offset};
                        MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                        i_derivative += -this->state.state["h1"](k) * sigmoid_i2(cov);
                    }
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
#pragma omp parallel for
            for (int i = 0; i < student_hidden; i++)
            {
                float i_derivative = 0.0;
#pragma omp parallel sections reduction(+ : i_derivative)
                {
#pragma omp section
                    for (int p = 0; p < teacher_hidden; p++)
                    {
                        std::vector<int> indices{i + input_noise_offset, teacher_2_offset + p};
                        MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                        i_derivative += this->state.state["th2"](p) * sigmoid_i2(cov);
                    }
#pragma omp section
                    for (int k = 0; k < student_hidden; k++)
                    {
                        std::vector<int> indices{i + input_noise_offset, k + input_noise_offset};
                        MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                        i_derivative -= this->state.state["h2"](k) * sigmoid_i2(cov);
                    }
                }
                derivative(i) = timestep * h_learning_rate * i_derivative;
            }
        }
        return derivative;
    }
};