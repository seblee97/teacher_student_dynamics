#include <vector>
#include <Eigen/Dense>
#include "../integrals.cpp"
#include <tuple>
#include <algorithm>
// #include <omp.h>

using Eigen::MatrixXd;

class HMMODE
{
public:
    HMMODEState &state;

    int teacher_hidden;
    int student_hidden;
    float delta;
    int latent_dimension;
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

    Matrix<double, 1, Dynamic> d_rho_1;
    Matrix<double, 1, Dynamic> d_rho_2;
    // Matrix<double, 1, Dynamic> rho;
    Matrix<double, 1, Dynamic> sigma_rho_1_term;
    Matrix<double, 1, Dynamic> sigma_rho_2_term;

    // ufu2 = b^2; fu2 = c in goldt;
    double b;
    double c;

    // float rho_min;
    // float rho_max;
    // float rho_interval;

    HMMODE(
        HMMODEState &ode_state,
        int t_hidden,
        int s_hidden,
        std::string f_activation,
        float delta_frac,
        int lat_dimension,
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
                                   latent_dimension(lat_dimension),
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
        d_rho_1.resize(1, latent_dimension);
        d_rho_2.resize(1, latent_dimension);
        sigma_rho_1_term.resize(1, latent_dimension);
        sigma_rho_2_term.resize(1, latent_dimension);
        // rho_min = pow(1 - pow(delta, 0.5), 2);
        // rho_max = pow(1 + pow(delta, 0.5), 2);
        // rho_interval = (rho_max - rho_min) / latent_dimension;
        if (f_activation == "scaled_erf")
        {
            b = 1. / pow(M_PI, 0.5); 
            c = 1. / 3;              
        } else if (f_activation == "sign")
        {
            b = pow(2. / M_PI, 0.5); 
            c = 1.;                
        }
        else if (f_activation == "linear")
        {
            b = 1.;
            c = 1.;
        }
        for (int bin = 0; bin < latent_dimension; bin++)
        {
            double rho_b_1 = this->state.state["rho_1"](bin);
            double rho_b_2 = this->state.state["rho_2"](bin);
            // std::cout << "rhob1111" << rho_b_1 << std::endl;
            d_rho_1(0, bin) = pow(b, 2) * rho_b_1 + delta * (c - pow(b, 2));
            d_rho_2(0, bin) = pow(b, 2) * rho_b_2 + delta * (c - pow(b, 2));
            sigma_rho_1_term(0, bin) = (c - pow(b, 2)) * rho_b_1 + pow(rho_b_1 * b, 2) / delta;
            sigma_rho_2_term(0, bin) = (c - pow(b, 2)) * rho_b_2 + pow(rho_b_2 * b, 2) / delta;
        }
        teacher_1_offset = student_hidden;
        teacher_2_offset = student_hidden + teacher_hidden;
        set_active_teacher(0);

        // integrate_order_parameter_densities_etc();

        for (auto const &[key, val] : this->state.state)
        {
            std::cout << key << ':' << std::endl // order parameter name
                      << val << std::endl;       // matrix
        }

        std::cout << "delta = P/N " << delta << std::endl;
        std::cout << "b " << b << std::endl;
        std::cout << "c " << c << std::endl;
        std::cout << "timestep " << timestep << std::endl;
        std::cerr << "delta = P/N " << delta << std::endl;
        std::cerr << "b " << b << std::endl;
        std::cerr << "c " << c << std::endl;
        std::cerr << "timestep " << timestep << std::endl;
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

    void integrate_order_parameter_densities_etc()
    {
        this->state.set_order_parameter("Sigma1", this->state.state["sigma_1_density"].rowwise().mean().reshaped<Eigen::RowMajor>(student_hidden, student_hidden));
        this->state.set_order_parameter("Q", (c - pow(b, 2)) * this->state.state["W"] + pow(b, 2) * this->state.state["Sigma1"]);
        this->state.set_order_parameter("R", b * this->state.state["r_density"].rowwise().mean().reshaped<Eigen::RowMajor>(student_hidden, teacher_hidden));
        this->state.set_order_parameter("U", b * this->state.state["u_density"].rowwise().mean().reshaped<Eigen::RowMajor>(student_hidden, teacher_hidden));
    }

    std::tuple<double, double> step(std::string step_order_parameter_paths = "")
    {   
        // std::cout << "Taking ODE Step" << std::endl;
        this->state.step_covariance_matrix();
        // std::cout << "Stepped Cov Matrix" << std::endl;

        double e1 = 0;
        double e2 = 0;

        e1 += error_1();
        e2 += error_2();

        // std::cout << "error 1: " << e1 << std::endl;
        // std::cout << "error 2: " << e2 << std::endl;

        // MatrixXd Q_delta = MatrixXd::Constant(this->state.state["Q"].rows(), this->state.state["Q"].cols(), 0.0);
        MatrixXd W_delta = MatrixXd::Constant(this->state.state["W"].rows(), this->state.state["W"].cols(), 0.0);
        // MatrixXd Sigma1_delta = MatrixXd::Constant(this->state.state["Sigma1"].rows(), this->state.state["Sigma1"].cols(), 0.0);
        // MatrixXd Sigma2_delta = MatrixXd::Constant(this->state.state["Sigma2"].rows(), this->state.state["Sigma2"].cols(), 0.0);
        MatrixXd r_delta = MatrixXd::Constant(this->state.state["r_density"].rows(), this->state.state["r_density"].cols(), 0.0);
        // MatrixXd r_delta2 = MatrixXd::Constant(this->state.state["r_density"].rows(), this->state.state["r_density"].cols(), 0.0);
        MatrixXd sigma_1_delta = MatrixXd::Constant(this->state.state["sigma_1_density"].rows(), this->state.state["sigma_1_density"].cols(), 0.0);
        // MatrixXd u_delta = MatrixXd::Constant(this->state.state["U"].rows(), this->state.state["U"].cols(), 0.0);
        MatrixXd h1_delta = MatrixXd::Constant(this->state.state["h1"].rows(), this->state.state["h1"].cols(), 0.0);
        // MatrixXd h2_delta = MatrixXd::Constant(this->state.state["h2"].rows(), this->state.state["h2"].cols(), 0.0);

        if (train_w_layer)
        {
            r_delta += timestep * dr_dt();
            // r_delta2 += timestep * dr2_dt();
            W_delta += timestep * dW_dt();
            sigma_1_delta += timestep * dsigma_1_dt();
        }
        if (train_h_layer)
        {
            h1_delta += timestep * dh1_dt();
            if (multi_head)
            {
            }
        }

        // for (int k=0; k < student_hidden; k++){
        //     for (int l=0; l < teacher_hidden; l++){
        //         std::cout << "r_delta: " << r_delta(k,l) << "r_delta2: " << r_delta2(k,l) << std::endl;
        //     }
        // }

        std::vector<std::string> states_read;

        if (step_order_parameter_paths != ""){
            states_read = this->state.read_state_from_file(step_order_parameter_paths);
        }

        std::cout << step_order_parameter_paths << std::endl;

        std::vector<std::string>::iterator state_read;

        for (int i = 0; i < states_read.size(); i++){
            std::cout << "STATES_REDA" << states_read[i] << std::endl;
        }

        state_read = std::find(states_read.begin(), states_read.end(), "r_density");
        if (state_read == states_read.end())
        {
            std::cout << "r_dens" << std::endl;
            this->state.step_order_parameter("r_density", r_delta);
        }
        state_read = std::find(states_read.begin(), states_read.end(), "W");
        if (state_read == states_read.end())
        {
            std::cout << "STEPPED_W" << std::endl;
            this->state.step_order_parameter("W", W_delta);
        }
        state_read = std::find(states_read.begin(), states_read.end(), "sigma_1_density");
        if (state_read == states_read.end())
        {
            std::cout << "sig1_dens" << std::endl;
            this->state.step_order_parameter("sigma_1_density", sigma_1_delta);
        }
        state_read = std::find(states_read.begin(), states_read.end(), "h1");
        if (state_read == states_read.end())
        {
            std::cout << "h1_stepped" << std::endl;
            this->state.step_order_parameter("h1", h1_delta);
        }
        
        integrate_order_parameter_densities_etc();

        // this->state.step_covariance_matrix();

        // this->state.integrate_order_parameter_density("Sigma1", "sigma_1_density");
        // this->state.integrate_order_parameter_density("R", "r_density");
        // this->state.integrate_order_parameter_density("U");
        // this->state.step_order_parameter("U", u_delta);

        if (multi_head)
        {
            // this->state.step_order_parameter("h2", h2_delta);
        }

        std::tuple<double, double> step_errors;
        step_errors = std::make_tuple(e1, e2);
        return step_errors;
    }

    double error_1()
    {
        double error = 0;
        {
            for (int k = 0; k < student_hidden; k++)
            {
                for (int l = 0; l < student_hidden; l++)
                {
                    std::vector<int> indices{k, l};
                    MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                    error += 0.5 * this->state.state["h1"](k) * this->state.state["h1"](l) * sigmoid_i2(cov);
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

    double error_2()
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

        double error = 0;
// #pragma omp parallel sections reduction(+ : error)
        {
// #pragma omp section
// #pragma omp parallel for collapse(2)
            for (int i = 0; i < student_hidden; i++)
            {
                for (int j = 0; j < student_hidden; j++)
                {
                    std::vector<int> indices{i, j};
                    MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                    error += 0.5 * head(i) * head(j) * sigmoid_i2(cov);
                }
            }
// #pragma omp section
// #pragma omp parallel for collapse(2)
            for (int p = 0; p < teacher_hidden; p++)
            {
                for (int q = 0; q < teacher_hidden; q++)
                {
                    std::vector<int> indices{teacher_2_offset + p, teacher_2_offset + q};
                    MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                    error += 0.5 * this->state.state["th2"](p) * this->state.state["th2"](q) * sigmoid_i2(cov);
                }
            }
// #pragma omp section
// #pragma omp parallel for collapse(2)
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

    // MatrixXd dr2_dt()
    // {
    //     // std::cout << "dR/dt" << std::endl;
    //     MatrixXd teacher_head(teacher_hidden, 1);
    //     MatrixXd student_head(student_hidden, 1);
    //     int offset;
    //     if (active_teacher == 0)
    //     {
    //         teacher_head << this->state.state["th1"];
    //         student_head << this->state.state["h1"];
    //         offset = teacher_1_offset;
    //     }
    //     else
    //     {
    //         teacher_head << this->state.state["th2"];
    //         offset = teacher_2_offset;
    //         if (multi_head)
    //         {
    //             student_head << this->state.state["h2"];
    //         }
    //         else
    //         {
    //             student_head << this->state.state["h1"];
    //         }
    //     }

    //     MatrixXd r_derivative = MatrixXd::Constant(this->state.state["r_density"].rows(), this->state.state["r_density"].cols(), 0.0);

    //     for (int k = 0; k < student_hidden; k++)
    //     {
    //         std::vector<int> kkk_indices{k, k, k};
    //         MatrixXd kkk_cov = this->state.generate_sub_covariance_matrix(kkk_indices);
    //         for (int m = 0; m < teacher_hidden; m++)
    //         {
    //             std::cout << "shape " << this->state.state["r_density"].rows() << this->state.state["r_density"].cols() << std::endl;
    //             // std::cout << "k t m " << k * teacher_hidden + m << std::endl;
    //             MatrixXd rkm = this->state.state["r_density"].row(k * teacher_hidden + m);
    //             // for (int pri = 0; pri < rkm.size(); pri++)
    //             // {
    //             //     std::cout << "k " << k << std::endl;
    //             //     std::cout << "m " << m << std::endl;
    //             //     std::cout << "pri " << pri << std::endl;
    //             //     std::cout << "rkm density bin " << rkm(pri) << std::endl;
    //             // }
    //             MatrixXd rkm_derivative = MatrixXd::Constant(rkm.rows(), rkm.cols(), 0.0);
    //             for (int j = 0; j < student_hidden; j++)
    //             {
    //                 if (j == k)
    //                     continue;
    //                 std::vector<int> kkj_indices{k, k, j};
    //                 std::vector<int> kjj_indices{k, j, j};
    //                 MatrixXd kkj_cov = this->state.generate_sub_covariance_matrix(kkj_indices);
    //                 MatrixXd kjj_cov = this->state.generate_sub_covariance_matrix(kjj_indices);
    //                 // term 1
    //                 double nom = student_head(j) * (this->state.state["Q"](j, j) * sigmoid_i3(kkj_cov) - this->state.state["Q"](k, j) * sigmoid_i3(kjj_cov));
    //                 double den = this->state.state["Q"](j, j) * this->state.state["Q"](k, k) - pow(this->state.state["Q"](k, j), 2);
    //                 // std::cout << "drho_shape " << d_rho.size() << d_rho.rows() << d_rho.cols() << std::endl;
    //                 // std::cout << "rkm_shape " << rkm.size() << rkm.rows() << rkm.cols() << std::endl;
    //                 rkm_derivative += d_rho_1.cwiseProduct(rkm) * (nom / den);
    //                 // term 2
    //                 MatrixXd rjm = this->state.state["r_density"].row(j * teacher_hidden + m);
    //                 nom = student_head(j) * (this->state.state["Q"](k, k) * sigmoid_i3(kjj_cov) - this->state.state["Q"](k, j) * sigmoid_i3(kkj_cov));
    //                 rkm_derivative += d_rho_1.cwiseProduct(rjm) * (nom / den);
    //             }
    //             // term 3
    //             rkm_derivative += student_head(k) * d_rho_1.cwiseProduct(rkm) * (sigmoid_i3(kkk_cov) / this->state.state["Q"](k, k));
    //             for (int n = 0; n < teacher_hidden; n++)
    //             {
    //                 std::vector<int> kkn_indices{k, k, n + offset};
    //                 std::vector<int> knn_indices{k, n + offset, n + offset};
    //                 MatrixXd kkn_cov = this->state.generate_sub_covariance_matrix(kkn_indices);
    //                 MatrixXd knn_cov = this->state.generate_sub_covariance_matrix(knn_indices);
    //                 // term 4
    //                 double nom = teacher_head(n) * (this->state.state["T"](n, n) * sigmoid_i3(kkn_cov) - this->state.state["R"](k, n) * sigmoid_i3(knn_cov));
    //                 double den = this->state.state["Q"](k, k) * this->state.state["T"](n, n) - pow(this->state.state["R"](k, n), 2);
    //                 rkm_derivative -= d_rho_1.cwiseProduct(rkm) * (nom / den);
    //                 // term 5
    //                 nom = teacher_head(n) * this->state.state["T_tilde"](n, m) * (this->state.state["Q"](k, k) * sigmoid_i3(knn_cov) - this->state.state["R"](k, n) * sigmoid_i3(kkn_cov));
    //                 rkm_derivative -= b * this->state.state["rho_1"] * (nom / den);
    //             }
    //             rkm_derivative *= (-1 * (w_learning_rate * student_head(k)) / delta);
    //             r_derivative.row(k * teacher_hidden + m) = rkm_derivative;
    //         }
    //     }
    //     return r_derivative;
    // }

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
                MatrixXd rkm_derivative = MatrixXd::Constant(rkm.rows(), rkm.cols(), 0.0);

                for (int j = 0; j < student_hidden; j++) {
                    if (j == k)
                        continue;
                    
                    double den = this->state.state["Q"](j, j) * this->state.state["Q"](k, k) - pow(this->state.state["Q"](k, j), 2);

                    // first line
                    std::vector<int> kkj_indices{k, k, j};
                    MatrixXd kkj_cov = this->state.generate_sub_covariance_matrix(kkj_indices);
                    rkm_derivative -= d_rho_1.cwiseProduct(rkm) * student_head(k) * student_head(j) * this->state.state["Q"](j, j) * sigmoid_i3(kkj_cov) / den;
                    std::vector<int> kjj_indices{k, j, j};
                    MatrixXd kjj_cov = this->state.generate_sub_covariance_matrix(kjj_indices);
                    rkm_derivative += d_rho_1.cwiseProduct(rkm) * student_head(k) * student_head(j) * this->state.state["Q"](k, j) * sigmoid_i3(kjj_cov) / den;

                    // second line
                    MatrixXd rjm = this->state.state["r_density"].row(j * teacher_hidden + m);
                    rkm_derivative -= d_rho_1.cwiseProduct(rjm) * student_head(k) * student_head(j) * this->state.state["Q"](k, k) * sigmoid_i3(kjj_cov) / den;
                    rkm_derivative += d_rho_1.cwiseProduct(rjm) * student_head(k) * student_head(j) * this->state.state["Q"](k, j) * sigmoid_i3(kkj_cov) / den;
                }

                // third line
                rkm_derivative -= d_rho_1.cwiseProduct(rkm) * student_head(k) * student_head(k) * sigmoid_i3(kkk_cov) / this->state.state["Q"](k, k);

                for (int n = 0; n < teacher_hidden; n++) {
                    double den = this->state.state["Q"](k, k) * this->state.state["T"](n, n) - pow(this->state.state["R"](k, n), 2);

                    // fourth line
                    std::vector<int> kkn_indices{k, k, n + offset};
                    MatrixXd kkn_cov = this->state.generate_sub_covariance_matrix(kkn_indices);
                    rkm_derivative += d_rho_1.cwiseProduct(rkm) * student_head(k) * teacher_head(n) * this->state.state["T"](n, n) * sigmoid_i3(kkn_cov) / den;

                    std::vector<int> knn_indices{k, n + offset, n + offset};
                    MatrixXd knn_cov = this->state.generate_sub_covariance_matrix(knn_indices);
                    rkm_derivative -= d_rho_1.cwiseProduct(rkm) * student_head(k) * teacher_head(n) * this->state.state["R"](k, n) * sigmoid_i3(knn_cov) / den;

                    // fifth line
                    rkm_derivative += b * this->state.state["rho_1"] * student_head(k) * teacher_head(n) * this->state.state["T_tilde"](n, m) * this->state.state["Q"](k, k) * sigmoid_i3(knn_cov) / den;
                    rkm_derivative -= b * this->state.state["rho_1"] * student_head(k) * teacher_head(n) * this->state.state["T_tilde"](n, m) * this->state.state["R"](k, n) * sigmoid_i3(kkn_cov) / den;
                }
                r_derivative.row(k * teacher_hidden + m) = w_learning_rate * rkm_derivative / delta;
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
            for (int l = k; l < student_hidden; l++)
            {
                double W_kl_derivative = 0;
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
                    std::vector<int> kln_indices{k, l, n + offset};
                    std::vector<int> lkn_indices{l, k, n + offset};
                    MatrixXd kln_cov = this->state.generate_sub_covariance_matrix(kln_indices);
                    MatrixXd lkn_cov = this->state.generate_sub_covariance_matrix(lkn_indices);
                    W_kl_derivative += w_learning_rate * teacher_head(n) * (student_head(k) * sigmoid_i3(kln_cov) + student_head(l) * sigmoid_i3(lkn_cov));
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
                        std::vector<int> kljm_indices{k, l, j, m + offset};
                        MatrixXd kljm_cov = this->state.generate_sub_covariance_matrix(kljm_indices);
                        W_kl_derivative -= 2 * c * pow(w_learning_rate, 2) * student_head(k) * student_head(l) * student_head(j) * teacher_head(m) * sigmoid_i4(kljm_cov);
                    }
                }
                for (int n = 0; n < teacher_hidden; n++)
                {
                    for (int m = 0; m < teacher_hidden; m++)
                    {
                        // term 3c.
                        std::vector<int> klnm_indices{k, l, n + offset, m + offset};
                        MatrixXd klnm_cov = this->state.generate_sub_covariance_matrix(klnm_indices);
                        W_kl_derivative += c * pow(w_learning_rate, 2) * student_head(k) * student_head(l) * teacher_head(n) * teacher_head(m) * sigmoid_i4(klnm_cov);
                    }
                }
                W_derivative(k, l) = W_kl_derivative;
            }
        }
        W_derivative.triangularView<Eigen::Lower>() = W_derivative.transpose();
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
                MatrixXd sigma_1_kl_derivative_1 = MatrixXd::Constant(1, latent_dimension, 0.0);
                MatrixXd sigma_1_kl_derivative_2 = MatrixXd::Constant(1, latent_dimension, 0.0);
                for (int j = 0; j < student_hidden; j++)
                {
                    if (j != k)
                    {
                        std::vector<int> kkj_indices{k, k, j};
                        std::vector<int> kjj_indices{k, j, j};
                        MatrixXd kkj_cov = this->state.generate_sub_covariance_matrix(kkj_indices);
                        MatrixXd kjj_cov = this->state.generate_sub_covariance_matrix(kjj_indices);
                        // term 1
                        double nom = student_head(j) * (this->state.state["Q"](j, j) * sigmoid_i3(kkj_cov) - this->state.state["Q"](k, j) * sigmoid_i3(kjj_cov));
                        double den = this->state.state["Q"](j, j) * this->state.state["Q"](k, k) - pow(this->state.state["Q"](k, j), 2);
                        sigma_1_kl_derivative_1 += student_head(k) * d_rho_1.cwiseProduct(sigma_1_kl) * (nom / den);
                        // term 2
                        MatrixXd sigma_1_jl = this->state.state["sigma_1_density"].row(j * student_hidden + l);
                        nom = student_head(j) * (this->state.state["Q"](k, k) * sigmoid_i3(kjj_cov) - this->state.state["Q"](k, j) * sigmoid_i3(kkj_cov));
                        sigma_1_kl_derivative_1 += d_rho_1.cwiseProduct(sigma_1_jl) * student_head(k) * (nom / den);
                    }
                }
                // term 3
                sigma_1_kl_derivative_1 += student_head(k) * d_rho_1.cwiseProduct(sigma_1_kl) * student_head(k) * (sigmoid_i3(kkk_cov) / this->state.state["Q"](k, k));
                for (int n = 0; n < teacher_hidden; n++)
                {
                    std::vector<int> kkn_indices{k, k, n + offset};
                    std::vector<int> knn_indices{k, n + offset, n + offset};
                    MatrixXd kkn_cov = this->state.generate_sub_covariance_matrix(kkn_indices);
                    MatrixXd knn_cov = this->state.generate_sub_covariance_matrix(knn_indices);
                    MatrixXd rln = this->state.state["r_density"].row(l * teacher_hidden + n);
                    // term 4
                    double nom = teacher_head(n) * (this->state.state["T"](n, n) * sigmoid_i3(kkn_cov) - this->state.state["R"](k, n) * sigmoid_i3(knn_cov));
                    double den = this->state.state["Q"](k, k) * this->state.state["T"](n, n) - pow(this->state.state["R"](k, n), 2);
                    sigma_1_kl_derivative_1 -= d_rho_1.cwiseProduct(sigma_1_kl) * student_head(k) * (nom / den);
                    // term 5
                    nom = teacher_head(n) * (this->state.state["Q"](k, k) * sigmoid_i3(knn_cov) - this->state.state["R"](k, n) * sigmoid_i3(kkn_cov));
                    sigma_1_kl_derivative_1 -= b * this->state.state["rho_1"].cwiseProduct(rln) * student_head(k) * (nom / den);
                }

                // invert with k -> l, l-> k
                MatrixXd sigma_1_lk = this->state.state["sigma_1_density"].row(l * student_hidden + k);
                for (int j = 0; j < student_hidden; j++)
                {
                    if (j != l)
                    {
                        std::vector<int> llj_indices{l, l, j};
                        std::vector<int> ljj_indices{l, j, j};
                        MatrixXd llj_cov = this->state.generate_sub_covariance_matrix(llj_indices);
                        MatrixXd ljj_cov = this->state.generate_sub_covariance_matrix(ljj_indices);
                        // term 1
                        double nom = student_head(j) * (this->state.state["Q"](j, j) * sigmoid_i3(llj_cov) - this->state.state["Q"](l, j) * sigmoid_i3(ljj_cov));
                        double den = this->state.state["Q"](j, j) * this->state.state["Q"](l, l) - pow(this->state.state["Q"](l, j), 2);
                        sigma_1_kl_derivative_1 += student_head(l) * d_rho_1.cwiseProduct(sigma_1_lk) * (nom / den);
                        // term 2
                        MatrixXd sigma_1_jk = this->state.state["sigma_1_density"].row(j * student_hidden + k);
                        nom = student_head(j) * (this->state.state["Q"](l, l) * sigmoid_i3(ljj_cov) - this->state.state["Q"](l, j) * sigmoid_i3(llj_cov));
                        sigma_1_kl_derivative_1 += d_rho_1.cwiseProduct(sigma_1_jk) * student_head(l) * (nom / den);
                    }
                }
                // term 3
                sigma_1_kl_derivative_1 += student_head(l) * d_rho_1.cwiseProduct(sigma_1_lk) * student_head(l) * (sigmoid_i3(lll_cov) / this->state.state["Q"](l, l));
                for (int n = 0; n < teacher_hidden; n++)
                {
                    std::vector<int> lln_indices{l, l, n + offset};
                    std::vector<int> lnn_indices{l, n + offset, n + offset};
                    MatrixXd lln_cov = this->state.generate_sub_covariance_matrix(lln_indices);
                    MatrixXd lnn_cov = this->state.generate_sub_covariance_matrix(lnn_indices);
                    MatrixXd rkn = this->state.state["r_density"].row(k * teacher_hidden + n);
                    // term 4
                    double nom = teacher_head(n) * (this->state.state["T"](n, n) * sigmoid_i3(lln_cov) - this->state.state["R"](l, n) * sigmoid_i3(lnn_cov));
                    double den = this->state.state["Q"](l, l) * this->state.state["T"](n, n) - pow(this->state.state["R"](l, n), 2);
                    sigma_1_kl_derivative_1 -= d_rho_1.cwiseProduct(sigma_1_lk) * student_head(l) * (nom / den);
                    // term 5
                    nom = teacher_head(n) * (this->state.state["Q"](l, l) * sigmoid_i3(lnn_cov) - this->state.state["R"](l, n) * sigmoid_i3(lln_cov));
                    sigma_1_kl_derivative_1 -= b * this->state.state["rho_1"].cwiseProduct(rkn) * student_head(l) * (nom / den);
                }
                sigma_1_kl_derivative_1 *= (-w_learning_rate / delta);

                // term 6
                for (int j = 0; j < student_hidden; j++)
                {
                    for (int y = 0; y < student_hidden; y++)
                    {
                        std::vector<int> kljy_indices{k, l, j, y};
                        MatrixXd kljy_cov = this->state.generate_sub_covariance_matrix(kljy_indices);
                        sigma_1_kl_derivative_2 += sigma_rho_1_term * student_head(j) * student_head(y) * sigmoid_i4(kljy_cov);
                    }
                    for (int m = 0; m < teacher_hidden; m++)
                    {
                        std::vector<int> kljm_indices{k, l, j, m + offset};
                        MatrixXd kljm_cov = this->state.generate_sub_covariance_matrix(kljm_indices);
                        sigma_1_kl_derivative_2 -= 2 * sigma_rho_1_term * student_head(j) * teacher_head(m) * sigmoid_i4(kljm_cov);
                    }
                }
                for (int n = 0; n < teacher_hidden; n++)
                {
                    for (int m = 0; m < teacher_hidden; m++)
                    {
                        std::vector<int> klnm_indices{k, l, n + offset, m + offset};
                        MatrixXd klnm_cov = this->state.generate_sub_covariance_matrix(klnm_indices);
                        sigma_1_kl_derivative_2 += sigma_rho_1_term * teacher_head(n) * teacher_head(m) * sigmoid_i4(klnm_cov);
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
        if (train_h_layer and ((active_teacher == 0) or (not multi_head)))
        {
// #pragma omp parallel for
            for (int k = 0; k < student_hidden; k++)
            {
                double k_derivative = 0.0;
// #pragma omp parallel sections reduction(+ : k_derivative)
                // {
// #pragma omp section
                for (int n = 0; n < teacher_hidden; n++)
                {
                    std::vector<int> indices{k, n + offset};
                    MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                    k_derivative += teacher_head(n) * sigmoid_i2(cov);
                }
// #pragma omp section
                for (int j = 0; j < student_hidden; j++)
                {
                    std::vector<int> indices{k, j};
                    MatrixXd cov = this->state.generate_sub_covariance_matrix(indices);
                    k_derivative -= this->state.state["h1"](j) * sigmoid_i2(cov);
                }
                // }
                derivative(k) = h_learning_rate * k_derivative;
            }
        }
        return derivative;
    }
};