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

        MatrixXd Q_delta = MatrixXd::Constant(this->state.state["Q"].rows(), this->state.state["Q"].cols(), 0.0);
        MatrixXd W_delta = MatrixXd::Constant(this->state.state["W"].rows(), this->state.state["W"].cols(), 0.0);
        MatrixXd sigma1_delta = MatrixXd::Constant(this->state.state["Sigma1"].rows(), this->state.state["Sigma1"].cols(), 0.0);
        MatrixXd sigma2_delta = MatrixXd::Constant(this->state.state["Sigma2"].rows(), this->state.state["Sigma2"].cols(), 0.0);
        MatrixXd r_delta = MatrixXd::Constant(this->state.state["r_density"].rows(), this->state.state["r_density"].cols(), 0.0);
        MatrixXd sigma_1_delta = MatrixXd::Constant(this->state.state["sigma_1_density"].rows(), this->state.state["sigma_1_density"].cols(), 0.0);
        MatrixXd u_delta = MatrixXd::Constant(this->state.state["U"].rows(), this->state.state["U"].cols(), 0.0);
        MatrixXd h1_delta = MatrixXd::Constant(this->state.state["h1"].rows(), this->state.state["h1"].cols(), 0.0);
        MatrixXd h2_delta = MatrixXd::Constant(this->state.state["h2"].rows(), this->state.state["h2"].cols(), 0.0);

        if (train_w_layer)
        {
            r_delta += timestep * dr_dt();
            W_delta += timestep * dW_dt();
            sigma_1_delta += timestep * dsigma_1_dt();
            h1_delta += timestep * dh1_dt();
        }
        if (train_h_layer)
        {
            if (multi_head)
            {
            }
        }

        this->state.step_order_parameter("W", W_delta);
        this->state.step_order_parameter("sigma_1_density", sigma_1_delta);
        this->state.step_order_parameter("r_density", r_delta);

        this->state.set_order_parameter("Q", (c - pow(b, 2)) * this->state.state["W"] + pow(b, 2) * this->state.state["Sigma1"]);
        this->state.set_order_parameter("R", b * this->state.state["r_density"].rowwise().mean().reshaped(student_hidden, teacher_hidden));
        this->state.set_order_parameter("Sigma1", this->state.state["sigma_1_density"].rowwise().mean().reshaped(student_hidden, student_hidden));

        // this->state.integrate_order_parameter_density("Sigma1", "sigma_1_density");
        // this->state.integrate_order_parameter_density("R", "r_density");
        // this->state.integrate_order_parameter_density("U");
        // this->state.step_order_parameter("U", u_delta);
        this->state.step_order_parameter("h1", h1_delta);

        if (multi_head)
        {
            // this->state.step_order_parameter("h2", h2_delta);
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
            for (int k = 0; k < student_hidden; k++)
            {
                for (int l = 0; l < student_hidden; l++)
                {
                    std::vector<int> indices{k, l};
                    MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                    error += 0.5 * this->state.state["h1"](k) * this->state.state["h1"](l) * sigmoid_i2(cov);
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
            for (int k = 0; k < student_hidden; k++)
            {
                for (int n = 0; n < teacher_hidden; n++)
                {
                    std::vector<int> indices{k, teacher_1_offset + n};
                    MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                    error -= this->state.state["h1"](k) * this->state.state["th1"](n) * sigmoid_i2(cov);
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

        for (int k = 0; k < student_hidden; k++)
        {
            std::vector<int> kkk_indices{k, k, k};
            MatrixXd kkk_cov = this->state.generate_sub_covariance_matrix(kkk_indices);
            for (int m = 0; m < teacher_hidden; m++)
            {
                std::cout << "shape " << this->state.state["r_density"].rows() << this->state.state["r_density"].cols() << std::endl;
                // std::cout << "k t m " << k * teacher_hidden + m << std::endl;
                MatrixXd rkm = this->state.state["r_density"].row(k * teacher_hidden + m);
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
                        // term 1
                        float nom = student_head(j) * (this->state.state["Q"](j, j) * sigmoid_i3(kkj_cov) - this->state.state["Q"](k, j) * sigmoid_i3(kjj_cov));
                        float den = this->state.state["Q"](j, j) * this->state.state["Q"](k, k) - pow(this->state.state["Q"](k, j), 2);
                        std::cout << "drho_shape " << d_rho.size() << d_rho.rows() << d_rho.cols() << std::endl;
                        std::cout << "rkm_shape " << rkm.size() << rkm.rows() << rkm.cols() << std::endl;
                        rkm_derivative += d_rho.cwiseProduct(rkm) * (nom / den);
                        // term 2
                        MatrixXd rjm = this->state.state["r_density"].row(j * teacher_hidden + m);
                        nom = student_head(j) * (this->state.state["Q"](k, k) * sigmoid_i3(kjj_cov) - this->state.state["Q"](k, j) * sigmoid_i3(kkj_cov));
                        rkm_derivative += d_rho.cwiseProduct(rjm) * (nom / den);
                    }
                }
                // term 3
                rkm_derivative += student_head(k) * d_rho.cwiseProduct(rkm) * (sigmoid_i3(kkk_cov) / this->state.state["Q"](k, k));
                for (int n = 0; n < teacher_hidden; n++)
                {
                    std::vector<int> kkn_indices{k, k, n};
                    std::vector<int> knn_indices{k, n, n};
                    MatrixXd kkn_cov = this->state.generate_sub_covariance_matrix(kkn_indices);
                    MatrixXd knn_cov = this->state.generate_sub_covariance_matrix(knn_indices);
                    // term 4
                    float nom = teacher_head(n) * (this->state.state["T"](n, n) * sigmoid_i3(kkn_cov) - this->state.state["R"](k, n) * sigmoid_i3(knn_cov));
                    float den = this->state.state["Q"](k, k) * this->state.state["T"](n, n) - pow(this->state.state["R"](k, n), 2);
                    rkm_derivative -= d_rho.cwiseProduct(rkm) * (nom / den);
                    // term 5
                    nom = teacher_head(n) * (this->state.state["T_tilde"](n, m) * this->state.state["Q"](k, k) * sigmoid_i3(knn_cov) - this->state.state["R"](k, n) * sigmoid_i3(kkn_cov));
                    rkm_derivative -= b * rho * (nom / den);
                }
                rkm_derivative *= (-(w_learning_rate * student_head(k)) / delta);
                r_derivative.row(k * teacher_hidden + m) = rkm_derivative;
            }
        }
        return r_derivative;
    }

    MatrixXd dW_dt()
    {
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

        MatrixXd W_derivative = MatrixXd::Constant(this->state.state["W"].rows(), this->state.state["W"].cols(), 0.0);

        for (int k = 0; k < student_hidden; k++)
        {
            for (int l = 0; l < student_hidden; l++)
            {
                float W_kl_derivative = 0;
                // first halves of terms 1 & 2
                for (int j = 0; j < student_hidden; j++)
                {
                    std::vector<int> klj_indices{k, l, j};
                    std::vector<int> lkj_indices{l, k, j};
                    MatrixXd klj_cov = this->state.generate_sub_covariance_matrix(klj_indices);
                    MatrixXd lkj_cov = this->state.generate_sub_covariance_matrix(lkj_indices);
                    W_kl_derivative -= w_learning_rate * student_head(j) * (student_head(k) * sigmoid_i3(klj_cov) + student_head(l) * sigmoid_i3(lkj_cov));
                }
                // second halves of terms 1 & 2
                for (int n = 0; n < teacher_hidden; n++)
                {
                    std::vector<int> kln_indices{k, l, n};
                    std::vector<int> lkn_indices{l, k, n};
                    MatrixXd kln_cov = this->state.generate_sub_covariance_matrix(kln_indices);
                    MatrixXd lkn_cov = this->state.generate_sub_covariance_matrix(lkn_indices);
                    W_kl_derivative -= w_learning_rate * teacher_head(n) * (student_head(k) * sigmoid_i3(kln_cov) + student_head(l) * sigmoid_i3(lkn_cov));
                }
                for (int j = 0; j < student_hidden; j++)
                {
                    for (int a = 0; a < student_hidden; a++)
                    {
                        // term 3a.
                        std::vector<int> klja_indices{k, l, j, a};
                        MatrixXd klja_cov = this->state.generate_sub_covariance_matrix(klja_indices);
                        W_kl_derivative += c * pow(w_learning_rate, 2) * student_head(k) * student_head(l) * student_head(j) * student_head(a) * sigmoid_i4(klja_cov);
                    }
                    for (int m = 0; m < teacher_hidden; m++)
                    {
                        // term 3b.
                        std::vector<int> kljm_indices{k, l, j, m};
                        MatrixXd kljm_cov = this->state.generate_sub_covariance_matrix(kljm_indices);
                        W_kl_derivative -= 2 * c * pow(w_learning_rate, 2) * student_head(k) * student_head(l) * student_head(j) * teacher_head(m) * sigmoid_i4(kljm_cov);
                    }
                }
                for (int m = 0; m < teacher_hidden; m++)
                {
                    for (int n = 0; n < teacher_hidden; n++)
                    {
                        // term 3c.
                        std::vector<int> klnm_indices{k, l, n, m};
                        MatrixXd klnm_cov = this->state.generate_sub_covariance_matrix(klnm_indices);
                        W_kl_derivative += c * pow(w_learning_rate, 2) * student_head(k) * student_head(l) * teacher_head(n) * teacher_head(m) * sigmoid_i4(klnm_cov);
                    }
                }
                W_derivative(k, l) = W_kl_derivative;
            }
        }
        return W_derivative;
    }

    MatrixXd dsigma_1_dt()
    {
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

        MatrixXd sigma_1_derivative = MatrixXd::Constant(this->state.state["sigma_1_density"].rows(), this->state.state["sigma_1_density"].cols(), 0.0);

        for (int k = 0; k < student_hidden; k++)
        {
            std::vector<int> kkk_indices{k, k, k};
            MatrixXd kkk_cov = this->state.generate_sub_covariance_matrix(kkk_indices);
            for (int l = 0; l < student_hidden; l++)
            {
                std::vector<int> lll_indices{l, l, l};
                MatrixXd lll_cov = this->state.generate_sub_covariance_matrix(lll_indices);
                MatrixXd sigma_1_kl = this->state.state["sigma_1_density"].row(k * student_hidden + l);
                MatrixXd sigma_1_lk = this->state.state["sigma_1_density"].row(l * student_hidden + k);
                MatrixXd sigma_1_kl_derivative_1 = MatrixXd::Constant(1, num_bins, 0.0);
                MatrixXd sigma_1_kl_derivative_2 = MatrixXd::Constant(1, num_bins, 0.0);
                for (int j = 0; j < student_hidden; j++)
                {
                    if (j != k)
                    {
                        std::vector<int> kkj_indices{k, k, j};
                        std::vector<int> kjj_indices{k, j, j};
                        MatrixXd kkj_cov = this->state.generate_sub_covariance_matrix(kkj_indices);
                        MatrixXd kjj_cov = this->state.generate_sub_covariance_matrix(kjj_indices);
                        // term 1
                        float nom = student_head(j) * (this->state.state["Q"](j, j) * sigmoid_i3(kkj_cov) - this->state.state["Q"](k, j) * sigmoid_i3(kjj_cov));
                        float den = this->state.state["Q"](j, j) * this->state.state["Q"](k, k) - pow(this->state.state["Q"](k, j), 2);
                        sigma_1_kl_derivative_1 += d_rho.cwiseProduct(sigma_1_kl) * (nom / den);
                        // term 2
                        MatrixXd sigma_1_jl = this->state.state["sigma_1_density"].row(j * student_hidden + l);
                        nom = student_head(j) * (this->state.state["Q"](k, k) * sigmoid_i3(kjj_cov) - this->state.state["Q"](k, j) * sigmoid_i3(kkj_cov));
                        sigma_1_kl_derivative_1 += d_rho.cwiseProduct(sigma_1_jl) * student_head(k) * (nom / den);
                    }
                }
                // term 3
                sigma_1_kl_derivative_1 += student_head(k) * d_rho.cwiseProduct(sigma_1_kl) * student_head(k) * (sigmoid_i3(kkk_cov) / this->state.state["Q"](k, k));
                for (int n = 0; n < teacher_hidden; n++)
                {
                    std::vector<int> kkn_indices{k, k, n};
                    std::vector<int> knn_indices{k, n, n};
                    MatrixXd kkn_cov = this->state.generate_sub_covariance_matrix(kkn_indices);
                    MatrixXd knn_cov = this->state.generate_sub_covariance_matrix(knn_indices);
                    MatrixXd rln = this->state.state["r_density"].row(l * teacher_hidden + n);
                    // term 4
                    float nom = teacher_head(n) * (this->state.state["T"](n, n) * sigmoid_i3(kkn_cov) - this->state.state["R"](k, n) * sigmoid_i3(knn_cov));
                    float den = this->state.state["Q"](k, k) * this->state.state["T"](n, n) - pow(this->state.state["R"](k, n), 2);
                    sigma_1_kl_derivative_1 -= d_rho.cwiseProduct(sigma_1_kl) * student_head(k) * (nom / den);
                    // term 5
                    nom = teacher_head(n) * (this->state.state["Q"](k, k) * sigmoid_i3(knn_cov) - this->state.state["R"](k, n) * sigmoid_i3(kkn_cov));
                    sigma_1_kl_derivative_1 -= b * rho.cwiseProduct(rln) * student_head(k) * (nom / den);
                }

                // invert with k -> l, l-> k
                for (int j = 0; j < student_hidden; j++)
                {
                    if (j != l)
                    {
                        std::vector<int> llj_indices{l, l, j};
                        std::vector<int> ljj_indices{l, j, j};
                        MatrixXd llj_cov = this->state.generate_sub_covariance_matrix(llj_indices);
                        MatrixXd ljj_cov = this->state.generate_sub_covariance_matrix(ljj_indices);
                        // term 1
                        float nom = student_head(j) * (this->state.state["Q"](j, j) * sigmoid_i3(llj_cov) - this->state.state["Q"](l, j) * sigmoid_i3(ljj_cov));
                        float den = this->state.state["Q"](j, j) * this->state.state["Q"](l, l) - pow(this->state.state["Q"](l, j), 2);
                        sigma_1_kl_derivative_1 += d_rho.cwiseProduct(sigma_1_lk) * (nom / den);
                        // term 2
                        MatrixXd sigma_1_jk = this->state.state["sigma_1_density"].row(j * student_hidden + k);
                        nom = student_head(j) * (this->state.state["Q"](l, l) * sigmoid_i3(ljj_cov) - this->state.state["Q"](l, j) * sigmoid_i3(llj_cov));
                        sigma_1_kl_derivative_1 += d_rho.cwiseProduct(sigma_1_jk) * student_head(l) * (nom / den);
                    }
                }
                // term 3
                sigma_1_kl_derivative_1 += student_head(l) * d_rho.cwiseProduct(sigma_1_lk) * student_head(l) * (sigmoid_i3(lll_cov) / this->state.state["Q"](l, l));
                for (int n = 0; n < teacher_hidden; n++)
                {
                    std::vector<int> lln_indices{l, l, n};
                    std::vector<int> lnn_indices{l, n, n};
                    MatrixXd lln_cov = this->state.generate_sub_covariance_matrix(lln_indices);
                    MatrixXd lnn_cov = this->state.generate_sub_covariance_matrix(lnn_indices);
                    MatrixXd rkn = this->state.state["r_density"].row(k * teacher_hidden + n);
                    // term 4
                    float nom = teacher_head(n) * (this->state.state["T"](n, n) * sigmoid_i3(lln_cov) - this->state.state["R"](l, n) * sigmoid_i3(lnn_cov));
                    float den = this->state.state["Q"](l, l) * this->state.state["T"](n, n) - pow(this->state.state["R"](l, n), 2);
                    sigma_1_kl_derivative_1 -= d_rho.cwiseProduct(sigma_1_lk) * student_head(l) * (nom / den);
                    // term 5
                    nom = teacher_head(n) * (this->state.state["Q"](l, l) * sigmoid_i3(lnn_cov) - this->state.state["R"](l, n) * sigmoid_i3(lln_cov));
                    sigma_1_kl_derivative_1 -= b * rho.cwiseProduct(rkn) * student_head(l) * (nom / den);
                }
                sigma_1_kl_derivative_1 *= (-w_learning_rate / delta);

                // term 6
                for (int j = 0; j < student_hidden; j++)
                {
                    for (int y = 0; y < student_hidden; y++)
                    {
                        std::vector<int> kljy_indices{k, l, j, y};
                        MatrixXd kljy_cov = this->state.generate_sub_covariance_matrix(kljy_indices);
                        sigma_1_kl_derivative_2 += sigma_rho_term * student_head(j) * student_head(y) * sigmoid_i4(kljy_cov);
                    }
                    for (int m = 0; m < teacher_hidden; m++)
                    {
                        std::vector<int> kljm_indices{k, l, j, m};
                        MatrixXd kljm_cov = this->state.generate_sub_covariance_matrix(kljm_indices);
                        sigma_1_kl_derivative_2 -= 2 * sigma_rho_term * student_head(j) * teacher_head(m) * sigmoid_i4(kljm_cov);
                    }
                }
                for (int n = 0; n < teacher_hidden; n++)
                {
                    for (int m = 0; m < teacher_hidden; m++)
                    {
                        std::vector<int> klnm_indices{k, l, n, m};
                        MatrixXd klnm_cov = this->state.generate_sub_covariance_matrix(klnm_indices);
                        sigma_1_kl_derivative_2 += sigma_rho_term * teacher_head(n) * teacher_head(m) * sigmoid_i4(klnm_cov);
                    }
                }
                sigma_1_kl_derivative_2 *= student_head(k) * student_head(l) * pow(w_learning_rate, 2);
                sigma_1_derivative.row(k * student_hidden + l) = sigma_1_kl_derivative_1 + sigma_1_kl_derivative_2;
            }
        }
        return sigma_1_derivative;
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
        MatrixXd derivative = MatrixXd::Constant(this->state.state["h1"].rows(), this->state.state["h1"].cols(), 0.0);
        if (train_h_layer and (active_teacher == 0) or (not multi_head))
        {
#pragma omp parallel for
            for (int k = 0; k < student_hidden; k++)
            {
                float k_derivative = 0.0;
#pragma omp parallel sections reduction(+ : i_derivative)
                {
#pragma omp section
                    for (int n = 0; n < teacher_hidden; n++)
                    {
                        std::vector<int> indices{k, n};
                        MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                        k_derivative += teacher_head(n) * sigmoid_i2(cov);
                    }
#pragma omp section
                    for (int j = 0; j < student_hidden; j++)
                    {
                        std::vector<int> indices{k, j};
                        MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                        k_derivative += -this->state.state["h1"](k) * sigmoid_i2(cov);
                    }
                }
                derivative(k) = h_learning_rate * k_derivative;
            }
        }
        return derivative;
    }
};