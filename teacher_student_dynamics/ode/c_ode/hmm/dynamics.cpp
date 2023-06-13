#include <vector>
#include <Eigen/Dense>
#include "../integrals.cpp"
#include <tuple>
// #include <omp.h>

using Eigen::MatrixXd;

class HMMODE
{
public:
    HMMODEState &state;

    int teacher_hidden;
    int student_hidden;
    float delta;
    int num_bins;
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

    Matrix<double, 1, Dynamic> d_rho;
    Matrix<double, 1, Dynamic> rho;
    Matrix<double, 1, Dynamic> sigma_rho_term;

    float rho_min;
    float rho_max;
    float rho_interval;

    float b = 1 / pow(M_PI, 0.5);
    float c = 1 / 3;

    HMMODE(
        HMMODEState &ode_state,
        int t_hidden,
        int s_hidden,
        float delta_frac,
        int num_b,
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
                                   delta(delta_frac),
                                   num_bins(num_b),
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
        // d_rho = (c - b^2)\delta + b^2\rho (hard-coded for scaled erf)
        d_rho.resize(1, num_bins);
        rho.resize(1, num_bins);
        sigma_rho_term.resize(1, num_bins);
        rho_min = pow(1 - pow(delta, 0.5), 2);
        rho_max = pow(1 + pow(delta, 0.5), 2);
        rho_interval = (rho_max - rho_min) / num_bins;
        float rho_b;
        for (int bin = 0; bin < num_bins; bin++)
        {
            rho_b = rho_min + bin * rho_interval;
            rho(0, bin) = rho_b;
            d_rho(0, bin) = (rho_b / M_PI) + delta * (1 / 3 - M_PI);
            sigma_rho_term(0, bin) = (c - pow(b, 2)) * rho_b + pow(rho_b * b, 2) / delta_frac;
        }
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

    std::tuple<float, float> step()
    {
        // std::cout << "Taking ODE Step" << std::endl;
        this->state.step_covariance_matrix();
        // std::cout << "Stepped Cov Matrix" << std::endl;

        float e1 = 0;
        float e2 = 0;

        e1 += error_1();
        e2 += error_2();

        // std::cout << "error 1: " << e1 << std::endl;
        // std::cout << "error 2: " << e2 << std::endl;

        MatrixXd q_delta = MatrixXd::Constant(this->state.state["Q"].rows(), this->state.state["Q"].cols(), 0.0);
        MatrixXd w_delta = MatrixXd::Constant(this->state.state["W"].rows(), this->state.state["W"].cols(), 0.0);
        MatrixXd sigma1_delta = MatrixXd::Constant(this->state.state["Sigma1"].rows(), this->state.state["Sigma1"].cols(), 0.0);
        MatrixXd sigma2_delta = MatrixXd::Constant(this->state.state["Sigma2"].rows(), this->state.state["Sigma2"].cols(), 0.0);
        MatrixXd r_delta = MatrixXd::Constant(this->state.state["r_density"].rows(), this->state.state["r_density"].cols(), 0.0);
        MatrixXd u_delta = MatrixXd::Constant(this->state.state["U"].rows(), this->state.state["U"].cols(), 0.0);
        MatrixXd h1_delta = MatrixXd::Constant(this->state.state["h1"].rows(), this->state.state["h1"].cols(), 0.0);
        MatrixXd h2_delta = MatrixXd::Constant(this->state.state["h2"].rows(), this->state.state["h2"].cols(), 0.0);

        if (train_w_layer)
        {
            r_delta += dr_dt();
        }
        if (train_h_layer)
        {
            if (multi_head)
            {
            }
        }

        this->state.step_order_parameter("Q", q_delta);
        this->state.step_order_parameter("W", w_delta);
        this->state.step_order_parameter("Sigma1", sigma1_delta);
        this->state.step_order_parameter("Sigma2", sigma2_delta);
        this->state.step_order_parameter("r_density", r_delta);
        this->state.integrate_order_parameter_density("R", "r_density");
        // this->state.integrate_order_parameter_density("U");
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

        MatrixXd r_derivative = MatrixXd::Constant(this->state.state["r_density"].rows(), this->state.state["r_density"].cols(), 0.0);

        for (int k = 0; k < teacher_hidden; k++)
        {
            for (int m = 0; m < student_hidden; m++)
            {
                std::cout << "shape " << this->state.state["r_density"].rows() << this->state.state["r_density"].cols() << std::endl;
                MatrixXd rkm = this->state.state["r_density"].row(k * student_hidden + m);
                // for (int pri = 0; pri < rkm.size(); pri++)
                // {
                //     std::cout << "k " << k << std::endl;
                //     std::cout << "m " << m << std::endl;
                //     std::cout << "pri " << pri << std::endl;
                //     std::cout << "rkm density bin " << rkm(pri) << std::endl;
                // }
                MatrixXd rkm_derivative = MatrixXd::Constant(1, num_bins, 0.0);
                for (int j = 0; j < student_hidden; j++)
                {
                    if (j != k)
                    {
                        std::vector<int> kkj_indices{k, k, j};
                        std::vector<int> kjj_indices{k, j, j};
                        MatrixXd kkj_cov = this->state.generate_sub_covariance_matrix(kkj_indices);
                        MatrixXd kjj_cov = this->state.generate_sub_covariance_matrix(kjj_indices);
                        float nom = student_head(j) * (this->state.state["Q"](j, j) * sigmoid_i3(kkj_cov) - this->state.state["Q"](k, j) * sigmoid_i3(kjj_cov));
                        // + student_head(k) * student_head(j) * rjm * (this->state.state["Q"](k, k) * sigmoid_i3(kjj_cov) - this->state.state["Q"](k, j) * sigmoid_i3(kkj_cov));
                        float den = this->state.state["Q"](j, j) * this->state.state["Q"](k, k) - pow(this->state.state["Q"](k, j), 2);
                        std::cout << "drho_shape " << d_rho.size() << d_rho.rows() << d_rho.cols() << std::endl;
                        std::cout << "rkm_shape " << rkm.size() << rkm.rows() << rkm.cols() << std::endl;
                        rkm_derivative += d_rho.cwiseProduct(rkm) * (nom / den);
                    }
                }
                // for (int n = 0; n < student_hidden; n++)
                // {
                //     std::vector<int> kkn_indices{k, k, n};
                //     std::vector<int> knn_indices{k, n, n};
                //     MatrixXd kkn_cov = this->state.generate_sub_covariance_matrix(kkn_indices);
                //     MatrixXd knn_cov = this->state.generate_sub_covariance_matrix(knn_indices);
                //     float nom = teacher_head(n) * (this->state.state["T"](n, n) * sigmoid_i3(kkn_cov) - this->state.state["R"](k, n) * sigmoid_i3(knn_cov) - brhodrho * teacher_head(n) * tilde_T(n, m) * (this->state.state["Q"](k, k) * sigmoid_i3(knn_cov) - this->state.state["R"](k, n) * sigmoid_i3(kkn_cov)));
                //     float den = this->state.state["Q"](k, k) * this->state.state["T"](n, n) - pow(this->state.state["R"](k, n), 2);
                //     km_derivative -= nom / den;
                // }
                // std::vector<int> kkk_indices{k, k, k};
                // MatrixXd kkk_cov = this->state.generate_sub_covariance_matrix(kkk_indices);
                // km_derivative += this->state.state["v"](k) * rkm * sigmoid_i3(kkk_cov) / this->state.state["Q"](k, k);

                // derivative(k, m) = -w_learning_rate * km_derivative / delta;
                r_derivative.row(k * student_hidden + m) = rkm_derivative;
            }
        }
        return r_derivative;
    }
};