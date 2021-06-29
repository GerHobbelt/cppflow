//
// Created by sergio on 16/05/19.
//

#include "cppflow/cppflow.h"

#include <numeric>
#include <iomanip>
#include <thread>
#include <iostream>
#include <iostream>
#include <sstream>
#include <filesystem>
#include <memory>
#include <vector>
#include <ThreadPool.h>
#include <condition_variable>
#include <mutex>
using namespace std;

#include <iostream>  // std::cout
#include <algorithm> // std::remove_copy_if
#include <vector>    // std::vector

void stringToken(const std::string &str, const char &token, std::vector<std::string> &output)
{
    int off = 0;
    auto index = str.find_first_of(token, 0);
    for (;;)
    {
        auto index = str.find_first_of(token, off);
        if (index == -1)
        {
            auto label = str.substr(off, str.size() - off);
            output.emplace_back(label);
            break;
        }
        auto label = str.substr(off, index - off);
        output.emplace_back(label);
        off += label.size() + 1;
    }
}

class Inference
{
public:
    Inference(const std::string &model, const std::string &dict) : m_model(new cppflow::model(model)), m_labels()
    {
        std::ifstream ifs(dict);
        std::ostringstream oss;
        oss << ifs.rdbuf();
        ifs.close();
        std::string text = oss.str();
        stringToken(text, '\n', m_labels);
    }

    void show()
    {
        auto opers = m_model->get_operations();
        for (auto name : opers)
        {
            std::cout << name << std::endl;
        }
    }

    std::string run(const std::string &base64)
    {

        std::vector<char> text;

        try
        {
            auto input = cppflow::decode_png(cppflow::decode_base64(base64));
            input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
            input = cppflow::div(input, float(255.0));
            input = cppflow::sub(input, float(0.5));
            input = cppflow::expand_dims(input, 0);
            auto output = (*m_model)(input);
            auto predicts = cppflow::arg_max(output, 2).get_data<int64_t>();

            for (auto i = 0; i < 18; i++)
            {
                predicts[i] += 1;
                if (predicts[i] != m_labels.size() && (!(i > 0 && predicts[i] == predicts[i - 1]) || (i > 1 && predicts[i] == predicts[i - 2])))
                {
                    text.emplace_back(m_labels[predicts[i]].data()[0]);
                }
            }
        }
        catch (std::exception &e)
        {
            return std::string(e.what());
        }
        return std::string(text.begin(), text.end());
    }

private:
    std::shared_ptr<cppflow::model> m_model;
    std::vector<std::string> m_labels;
};

class InferencePool
{
public:
    InferencePool(int count) : m_model_pool(count), m_pool(count), m_id(0), m_mutex()
    {
    }

    void ready(const std::string &model, const std::string &dict)
    {
    }

    int run(const std::string &base64)
    {
        int id = 0;
        {
            std::scoped_lock sc(m_mutex);
            m_id += 1;
            id = m_id;
        }

        m_pool.enqueue([]() {

        });
        return id;
    }

private:
    std::vector<std::shared_ptr<Inference>> m_model_pool;
    ThreadPool m_pool;
    std::mutex m_mutex;
    int m_id;
};

Inference *inference;

#ifdef WIN32
extern "C"
{
    _declspec(dllexport) void load(char *model, char *dict)
    {
        printf("%s\n", model);
        inference = new Inference(std::string(model), std::string(dict));
    }

    _declspec(dllexport) void run(const char *base64, char *output)
    {
        auto result = inference->run(std::string(base64));

        memcpy(output, result.c_str(), result.size());

        return;
    }
}
#else
extern "C"
{
    void load(char *model, char *dict)
    {
        inference = new Inference(std::string(model), std::string(dict));
    }

    void run(const char *base64, char *output)
    {
        auto result = inference->run(std::string(base64));
        memcpy(output, result.c_str(), result.size());
        return;
    }
}
#endif
