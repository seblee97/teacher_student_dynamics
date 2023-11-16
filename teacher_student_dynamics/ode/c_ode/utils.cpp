#include <string>
#include <map>
#include <any>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <iterator>
#include <stdexcept>
#include <variant>
#include <chrono>

template <typename Out>
void split(const std::string &s, char delim, Out result)
{
    std::istringstream iss(s);
    std::string item;
    while (std::getline(iss, item, delim))
    {
        *result++ = item;
    }
}

std::vector<std::string> split(const std::string &s, char delim)
{
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}

// std::variant<int, float, std::string, bool, std::vector<float>> map_values_var

std::map<std::string, std::variant<int, float, std::string, bool, std::vector<float>, std::vector<int>, std::vector<std::string>>>
parse_input(std::string input_file_path)
{
    std::map<std::string, std::variant<int, float, std::string, bool, std::vector<float>, std::vector<int>, std::vector<std::string>>> config;

    std::ifstream ifs(input_file_path); // input file stream
    std::string line;

    if (ifs.is_open())
    {
        while (getline(ifs, line))
        {
            std::vector<std::string> split_string = split(line, ';');
            std::variant<int, float, std::string, bool, std::vector<float>, std::vector<int>, std::vector<std::string>> value;

            if (split_string[0] == "int")
            {
                value = stoi(split_string[2]);
            }
            else if (split_string[0] == "float")
            {
                value = std::stof(split_string[2]);
            }
            else if (split_string[0] == "str")
            {
                value = split_string[2];
            }
            else if (split_string[0] == "bool")
            {
                value = bool(std::stoi(split_string[2]));
            }
            // else if (split_string[0] == "it_empty")
            // {
            //     value = false;
            // } 
            else if (split_string[0] == "it_float")
            {
                std::vector<float> value_vector;
                std::vector<std::string> split_it = split(split_string[2], ',');

                for (int i = 0; i < split_it.size(); i++)
                {
                    float split_i = std::stof(split_it[i]);
                    value_vector.push_back(split_i);
                }
                value = value_vector;
            }
            else if (split_string[0] == "it_int")
            {
                std::vector<int> value_vector;
                std::vector<std::string> split_it = split(split_string[2], ',');

                for (int i = 0; i < split_it.size(); i++)
                {
                    int split_i = std::stoi(split_it[i]);
                    value_vector.push_back(split_i);
                }
                value = value_vector;
            }
            else if (split_string[0] == "it_str")
            {
                std::vector<std::string> value_vector;
                std::vector<std::string> split_it = split(split_string[2], ',');

                for (int i = 0; i < split_it.size(); i++)
                {
                    std::string split_i = split_it[i];
                    value_vector.push_back(split_i);
                }
                value = value_vector;
            }
            else
            {
                throw std::invalid_argument("type not recognised.\n");
            }
            config[split_string[1]] = value;
            {
                std::cout << "type " << split_string[0] << " key " << split_string[1] << " value " << split_string[2] << " type " << typeid(value).name() << std::endl;
            }
        }
        ifs.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }
    return config;
}

template <
    class clock_t = std::chrono::steady_clock,
    class duration_t = std::chrono::milliseconds>
auto since(std::chrono::time_point<clock_t, duration_t> const &start)
{
    return (clock_t::now() - start) / (float)1e9;
}