#include "state.cpp"
#include "dynamics.cpp"
#include "../utils.cpp"
#include <fstream>
#include <filesystem>
#include <chrono>
// #include <format>
#include <math.h>

int main(int argc, char **argv)
{
    std::string input_file_path = argv[1];
    std::map<std::string, std::variant<int, float, std::string, bool, std::vector<float>, std::vector<int>, std::vector<std::string>>> config;
    config = parse_input(input_file_path);

    // config element "order_parameter_paths" is path to txt file
    // with paths of saved order parameter matrices.
    // format of file should be:
    //  Q,/path/to/Q.csv
    //  R,/path/to/R.csv
    //  etc.
    std::string order_parameter_paths;
    std::string output_path_str;
    std::string experiment_log_path;
    int stdout_frequency;
    order_parameter_paths = std::get<std::string>(config["order_parameter_paths"]);
    output_path_str = std::get<std::string>(config["output_path"]);
    experiment_log_path = std::get<std::string>(config["stdout_path"]);
    stdout_frequency = std::get<int>(config["stdout_frequency"]);

    // set output path variable and pipe cout to file.
    std::filesystem::path output_path(output_path_str);
    std::ofstream out(experiment_log_path);
    std::cout.rdbuf(out.rdbuf());

    // set number of threads for use in parallelisation
    // int omp_num_threads;
    // omp_num_threads = std::get<int>(config["omp_num_threads"]);
    // if (omp_num_threads > 0)
    // {
    //     omp_set_num_threads(omp_num_threads);
    //     std::cout << "OMP Threads: " << omp_num_threads << std::endl;
    // }

    int num_steps = std::get<int>(config["num_steps"]);
    int log_frequency = std::get<int>(config["ode_log_frequency"]);
    int switch_step = std::get<int>(config["switch_step"]);
    int input_dimension = std::get<int>(config["input_dimension"]);
    int teacher_hidden = std::get<int>(config["teacher_hidden"]);
    int student_hidden = std::get<int>(config["student_hidden"]);
    bool multi_head = std::get<bool>(config["multi_head"]);
    float w_learning_rate = std::get<float>(config["hidden_learning_rate"]);
    float h_learning_rate = std::get<float>(config["head_learning_rate"]);
    float timestep = std::get<float>(config["timestep"]);
    bool train_w_layer = std::get<bool>(config["train_hidden_layer"]);
    bool train_h_layer = std::get<bool>(config["train_head_layer"]);
    std::vector<float> noise_stds = std::get<std::vector<float>>(config["noise_stds"]);
    std::vector<float> input_noise_stds = std::get<std::vector<float>>(config["input_noise_stds"]);
    std::vector<int> freeze_units = std::get<std::vector<int>>(config["freeze_units"]);

    std::cout << "configuration parsed successfully." << std::endl;

    ODEState state(teacher_hidden, student_hidden, multi_head, order_parameter_paths);

    for (auto const &[key, val] : state.state)
    {
        std::cout << key << ':' << std::endl // order parameter name
                  << val << std::endl;       // matrix
    }

    std::cout << "configuration covariance matrix:" << std::endl;
    std::cout << state.getCovarianceMatrix() << std::endl;

    float log_time = (float)log_frequency / (float)input_dimension;
    int log_step = static_cast<int>(std::round(log_time / timestep));

    std::cout << "log freq: " << log_frequency << std::endl;
    std::cout << "log time: " << log_time << std::endl;
    std::cout << "log step: " << log_step << std::endl;

    float time = (float)num_steps / (float)input_dimension;
    float switch_time = (float)switch_step / (float)input_dimension;
    int num_deltas = static_cast<int>(std::round(time / timestep));
    int num_logs = static_cast<int>(std::round(num_deltas / log_step));
    int switch_delta = static_cast<int>(std::round(switch_time / timestep));
    float step_scaling = input_dimension / (1 / timestep);

    std::cout << "num steps: " << num_steps << std::endl;
    std::cout << "input dimension: " << input_dimension << std::endl;
    std::cout << "time: " << time << std::endl;
    std::cout << "num deltas: " << num_deltas << std::endl;
    std::cout << "num logs: " << num_logs << std::endl;
    std::cout << "switch_delta: " << switch_delta << std::endl;
    std::cout << "step_scaling: " << step_scaling << std::endl;

    StudentTeacherODE ODE(
        state,
        teacher_hidden,
        student_hidden,
        multi_head,
        w_learning_rate,
        h_learning_rate,
        timestep,
        train_w_layer,
        train_h_layer,
        input_noise_stds,
        noise_stds,
        freeze_units);

    std::vector<int> teacher_index_log(num_logs);
    std::vector<double> error_0_log(num_logs);
    std::vector<double> error_1_log(num_logs);

    std::map<std::string, std::vector<double>> q_log_map;
    std::map<std::string, std::vector<double>> r_log_map;
    std::map<std::string, std::vector<double>> u_log_map;
    std::map<std::string, std::vector<double>> h_0_log_map;
    std::map<std::string, std::vector<double>> h_1_log_map;

    for (int i = 0; i < student_hidden; i++)
    {
        for (int j = 0; j < student_hidden; j++)
        {
            std::vector<double> q_ij_log(num_logs);
            q_log_map["q_" + std::to_string(i) + std::to_string(j)] = q_ij_log;
        }
    }

    for (int i = 0; i < student_hidden; i++)
    {
        for (int j = 0; j < teacher_hidden; j++)
        {
            std::vector<double> r_ij_log(num_logs);
            std::vector<double> u_ij_log(num_logs);
            r_log_map["r_" + std::to_string(i) + std::to_string(j)] = r_ij_log;
            u_log_map["u_" + std::to_string(i) + std::to_string(j)] = u_ij_log;
        }
    }

    for (int i = 0; i < student_hidden; i++)
    {
        std::vector<double> h_0i_log(num_logs);
        std::vector<double> h_1i_log(num_logs);
        h_0_log_map["h_0" + std::to_string(i)] = h_0i_log;
        h_1_log_map["h_1" + std::to_string(i)] = h_1i_log;
    }

    std::tuple<float, float> step_errors;

    int log_i = 0;

    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

    for (int i = 0; i < num_deltas; i++)
    {
        if (i == switch_delta)
        {
            std::cout << "Switching Teacher..." << std::endl;
            std::cerr << "Switching Teacher..." << std::endl;
            ODE.set_active_teacher(1);
        }
        if (i % stdout_frequency == 0)
        {
            std::cout << "Step: " << step_scaling * i << "; Elapsed (s): " << since(start_time).count() << std::endl;
            std::cerr << "Step: " << step_scaling * i << "; Elapsed (s): " << since(start_time).count() << std::endl;
            start_time = std::chrono::steady_clock::now();
        }
        step_errors = ODE.step();

        if (i % log_step == 0)
        {
            teacher_index_log[log_i] = ODE.get_active_teacher();
            error_0_log[log_i] = std::get<0>(step_errors);
            error_1_log[log_i] = std::get<1>(step_errors);

            for (int s = 0; s < student_hidden; s++)
            {
                for (int s_ = 0; s_ < student_hidden; s_++)
                {
                    q_log_map["q_" + std::to_string(s) + std::to_string(s_)][log_i] = state.state["Q"](s, s_);
                }
            }

            for (int s = 0; s < student_hidden; s++)
            {
                for (int t = 0; t < teacher_hidden; t++)
                {
                    r_log_map["r_" + std::to_string(s) + std::to_string(t)][log_i] = state.state["R"](s, t);
                    u_log_map["u_" + std::to_string(s) + std::to_string(t)][log_i] = state.state["U"](s, t);
                }
            }

            for (int s = 0; s < student_hidden; s++)
            {
                h_0_log_map["h_0" + std::to_string(s)][log_i] = state.state["h1"](s);
                if (multi_head)
                {
                    h_1_log_map["h_1" + std::to_string(s)][log_i] = state.state["h2"](s);
                }
            }
            log_i++;
        }

        // q_00_log[log_i] = state.state["Q"](0, 0);
        // q_01_log[log_i] = state.state["Q"](0, 1);
        // q_10_log[log_i] = state.state["Q"](1, 0);
        // q_11_log[log_i] = state.state["Q"](1, 1);

        // r_00_log[log_i] = state.state["R"](0, 0);
        // r_01_log[log_i] = state.state["R"](1, 0);

        // u_00_log[log_i] = state.state["U"](0, 0);
        // u_01_log[log_i] = state.state["U"](1, 0);

        // h_0_00_log[log_i] = state.state["h1"](0);
        // h_0_01_log[log_i] = state.state["h1"](1);

        // h_1_00_log[log_i] = state.state["h2"](0);
        // h_1_01_log[log_i] = state.state["h2"](1);
    }

    std::cout << "Solve complete, saving data..." << std::endl;

    std::filesystem::path log_csvs_path = output_path / "log_csvs";
    std::filesystem::create_directory(log_csvs_path);

    std::ofstream file;
    file.open(log_csvs_path / "teacher_index.csv");
    for (int i = 0; i < teacher_index_log.size(); i++)
    {
        file << teacher_index_log[i];
        if (i != teacher_index_log.size() - 1)
        {
            file << "\n";
        }
    }
    file.close();

    file.open(log_csvs_path / "generalisation_error_0.csv");
    for (int i = 0; i < error_0_log.size(); i++)
    {
        file << error_0_log[i];
        if (i != error_0_log.size() - 1)
        {
            file << "\n";
        }
    }
    file.close();

    file.open(log_csvs_path / "generalisation_error_1.csv");
    for (int i = 0; i < error_1_log.size(); i++)
    {
        file << error_1_log[i];
        if (i < error_1_log.size() - 1)
        {
            file << "\n";
        }
    }
    file.close();

    file.open(log_csvs_path / "log_generalisation_error_0.csv");
    for (int i = 0; i < error_0_log.size(); i++)
    {
        file << log10(error_0_log[i]);
        if (i != error_0_log.size() - 1)
        {
            file << "\n";
        }
    }
    file.close();

    file.open(log_csvs_path / "log_generalisation_error_1.csv");
    for (int i = 0; i < error_1_log.size(); i++)
    {
        file << log10(error_1_log[i]);
        if (i < error_1_log.size() - 1)
        {
            file << "\n";
        }
    }
    file.close();

    std::string csv_name;

    std::cout << "Q" << std::endl;
    for (int i = 0; i < student_hidden; i++)
    {
        for (int j = 0; j < student_hidden; j++)
        {
            csv_name = "student_self_overlap_" + std::to_string(i) + "_" + std::to_string(j) + ".csv";
            file.open(log_csvs_path / csv_name);
            for (int n = 0; n < num_logs; n++)
            {
                file << q_log_map["q_" + std::to_string(i) + std::to_string(j)][n];
                if (n < num_logs - 1)
                {
                    file << "\n";
                }
            }
            file.close();
        }
    }

    std::cout << "U / R" << std::endl;
    for (int i = 0; i < student_hidden; i++)
    {
        for (int j = 0; j < teacher_hidden; j++)
        {
            csv_name = "student_teacher_1_" + std::to_string(i) + "_" + std::to_string(j) + ".csv";
            file.open(log_csvs_path / csv_name);
            for (int n = 0; n < num_logs; n++)
            {
                file << u_log_map["u_" + std::to_string(i) + std::to_string(j)][n];
                if (n < num_logs - 1)
                {
                    file << "\n";
                }
            }
            file.close();
            csv_name = "student_teacher_0_" + std::to_string(i) + "_" + std::to_string(j) + ".csv";
            file.open(log_csvs_path / csv_name);
            for (int n = 0; n < num_logs; n++)
            {
                file << r_log_map["r_" + std::to_string(i) + std::to_string(j)][n];
                if (n < num_logs - 1)
                {
                    file << "\n";
                }
            }
            file.close();
        }
    }

    std::cout << "h" << std::endl;
    std::cout << student_hidden << std::endl;
    for (int i = 0; i < student_hidden; i++)
    {
        csv_name = "student_head_0_weight_" + std::to_string(i) + ".csv";
        file.open(log_csvs_path / csv_name);
        for (int n = 0; n < num_logs; n++)
        {
            file << h_0_log_map["h_0" + std::to_string(i)][n];
            if (n < num_logs - 1)
            {
                file << "\n";
            }
        }
        file.close();
        if (multi_head)
        {
            csv_name = "student_head_1_weight_" + std::to_string(i) + ".csv";
            file.open(log_csvs_path / csv_name);
            for (int n = 0; n < num_logs; n++)
            {
                file << h_1_log_map["h_1" + std::to_string(i)][n];
                if (n < num_logs - 1)
                {
                    file << "\n";
                }
            }
            file.close();
        }
    }
}
