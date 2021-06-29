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
#include "httplib.h"
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
    Inference(const std::string &model, const std::string &dict) : m_model(new cppflow::model(model, true)), m_labels()
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
        show();
        std::string text;
        try
        {

            auto input = cppflow::decode_png(cppflow::read_file(base64));
            input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
            input = cppflow::expand_dims(input, 0);

            input = cppflow::resize_bicubic(input, {32, 43});

            input = cppflow::div(input, float(255.0));
            input = cppflow::sub(input, float(0.5));

            auto output = (*m_model)({{"input_1", input}}, {"Identity"})[0];
            std::cout << __LINE__ << "   xx  " << output.shape() << std::endl;
            auto data = output.get_data<float_t>();
            auto predicts = cppflow::arg_max(output, 0).get_data<int64_t>();
            auto max = 0.0;
            auto nclass = m_labels.size();
            std::vector<int> chars;
            std::vector<double> scores;

            auto width = output.shape().get_data<int64_t>()[2];
            auto height = output.shape().get_data<int64_t>()[1];

            std::cout << height << " " << width << " " << nclass << std::endl;
            for (int i = 0; i < height; i++)
            {
                int index = m_labels.size();
                for (int j = 0; j < width; j++)
                {
                    std::cout << " " << data[i * nclass + j];
                    if (data[i * nclass + j] >= max)
                    {
                        max = data[i * nclass + j];
                        index = j;
                    }
                }
                chars.push_back(index);
                scores.push_back(max);
                std::cout << std::endl;
                std::cout << index << " " << max << std::endl;
                max = 0.0;
            }

            double confidence = 0.0;
            double minConf = 0.1, maxConf = 1.0;
            int valid = 0;
            for (int i = 0; i < chars.size(); i++)
            {
                std::cout << chars[i] << std::endl;

                if (chars[i] != nclass - 1 &&
                    ((!(i > 0 && chars[i] == chars[i - 1])) || (i > 1 && chars[i] == chars[i - 2])))
                {
                    std::string value = m_labels[chars[i]];
                    if (scores[i] < minConf || scores[i] > maxConf)
                    {
                        continue;
                    }
                    text += value;
                }
            }
            std::cout << "text : " << text << std::endl;
        }
        catch (std::exception &e)
        {
            std::cout << "-----" << e.what() << std::endl;
            return std::string(e.what());
        }
        return text;
    }

private:
    std::shared_ptr<cppflow::model> m_model;
    std::vector<std::string> m_labels;
};

void http_server()
{
    using namespace httplib;

    Server svr;
    svr.new_task_queue = []
    { return new ThreadPool(12); };

    auto inference = new Inference(std::string("./5cf78304adce48a0cff43c3f.pb"), std::string("./dict.txt"));
    std::cout << __LINE__ << std::endl;
    inference->run("44.png");
    std::cout << __LINE__ << std::endl;

    svr.Get("/verify", [&](const Request &req, Response &res)
            {
                std::string value("xxx");
                if (req.has_param("captcha"))
                {
                    auto val = req.get_param_value("captcha");
                    value = inference->run(val);
                }
                res.set_content(value, "text/plain");
            });

    svr.Get("/stop", [&](const Request &req, Response &res)
            { svr.stop(); });

    svr.listen("0.0.0.0", 8080);
}
int main()
{
    http_server();
    // remove_copy_if example

    cppflow::model model("./");
    auto opers = model.get_operations();
    for (auto name : opers)
    {
        std::cout << name << std::endl;
    }

    std::ifstream ifs("./dict.txt");
    std::ostringstream oss;
    oss << ifs.rdbuf();
    ifs.close();

    std::string text = oss.str();
    std::vector<std::string> lables = {};
    stringToken(text, '\n', lables);
    for (auto label : lables)
    {
        std::cout << label << std::endl;
    }

    std::filesystem::directory_iterator list(std::filesystem::path("./"));
    for (auto &iter : list)
    {
        auto name = iter.path().filename().string();
        if (name.find(".png") == std::string::npos)
        {
            continue;
        }
        std::cout << name << std::endl;

        auto input = cppflow::decode_jpeg(cppflow::read_file(name));

        std::cout << opers[0] << std::endl;

        input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
        input = cppflow::div(input, float(255.0));
        input = cppflow::sub(input, float(0.5));
        input = cppflow::expand_dims(input, 0);

        try
        {
            auto output = model(input);
            auto predicts = cppflow::arg_max(output, 2).get_data<int64_t>();

            for (auto i = 0; i < 18; i++)
            {
                predicts[i] += 1;
                if (predicts[i] != lables.size() && (!(i > 0 && predicts[i] == predicts[i - 1]) || (i > 1 && predicts[i] == predicts[i - 2])))
                {
                    std::cout << lables[predicts[i]] << std::endl;
                }
            }
        }
        catch (std::exception &e)
        {
        }
    }

    // Read image
    /*  cv::Mat img = cv::imread("2.png", cv::IMREAD_COLOR);
    int rows = img.rows;
    int cols = img.cols;
    int channels = img.channels();

    // Put image in tensor
    std::vector<uint8_t> img_data;
    img_data.assign(img.data, img.data + img.total() * channels);
    auto input = cppflow::tensor(img_data, {rows, cols, channels});*/

    //  std::cout << "It's a tiger cat: " << cppflow::arg_max(output, 1) << std::endl;

    /*
    auto inpName = new cppflow::tensor(model, opers[0]);
    // Iterate through the operations of a graph.  To use:
    std::cout << __LINE__ << std::endl;

    cv::Mat img,
        inp;
    img = cv::imread("./1.png");
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    int rows = img.rows;
    int cols = img.cols;
    std::cout << __LINE__ << std::endl;

    auto dims1 = inpName->dim();

    cv::resize(img, inp, cv::Size(int(cols / (rows / 32.0f)), 32));

    cv::Mat fake_mat(inp.rows, inp.cols, CV_32FC(inp.channels()));
    inp.convertTo(fake_mat, CV_32FC(inp.channels()));

    fake_mat /= 255.0f;
    add(fake_mat, fake_mat, -0.5f);

    std::cout << fake_mat << std::endl;
    // Put image in Tensor
    std::vector<float> img_data = convertMat2Vector<float>(fake_mat);

    inpName->set_data(img_data, {1, inp.rows, inp.cols, 3});

    std::ifstream ifs("./1.txt");
    std::vector<std::string> labels_;
    for (auto i = 0; i < 1; i++)
    {
        std::thread th([&]() {
            double start = static_cast<double>(cv::getTickCount());
            auto outNames1 = new Tensor(model, opers[opers.size() - 1]);
            std::cout << opers[opers.size() - 1] << std::endl;
            model.run(inpName, {outNames1});
            std::vector<float> data = outNames1->get_data<float>();
            int nclass = 5990;
            float max = 0.0;
            std::vector<int> chars;
            std::vector<double> scores;
            double time = ((double)cv::getTickCount() - start) / cv::getTickFrequency();
            for (auto index : outNames1->dim())
            {
                std::cout << "index : " << index << std::endl;
            }
            for (int i = 0; i < 12; i++)
            {
                int index = 5990 - 1;
                for (int j = 0; j < 5990; j++)
                {

                    if (data[i * nclass + j] >= max)
                    {
                        max = data[i * nclass + j];
                        index = j;
                    }
                }
                chars.push_back(index);
                scores.push_back(max);
                max = 0.0;
            }

            string add_str;
            while (ifs)
            {
                getline(ifs, add_str);
                labels_.push_back(add_str);
            }
            ifs.close();
            int x = 0;
            std::string value = "";
            for (int i = 0; i < chars.size(); i++)
            {
                std::cout << chars[i] << "  " << scores[i] << std::endl;
                if (chars[i] != nclass - 1 &&
                    ((!(i > 0 && chars[i] == chars[i - 1])) || (i > 1 && chars[i] == chars[i - 2])))
                {
                    std::string value1 = labels_[chars[i] + 1];
                    if (scores[i] < 0.5)
                    {

                        continue;
                    }
                    std::cout << "value : " << value1 << std::endl;
                    value += value1;
                }
            }
            std::cout << value << std::endl;
            std::cout << "time : " << time << std::endl;

            auto dims = outNames1->dim();
            for (auto value : dims)
            {
                std::cout << "value: " << value << std::endl;
            }
        });
        th.detach();
    }
    cv::waitKey(1000 * 15);
    for (auto i = 0; i < 10; i++)
    {
        std::thread th([&]() {
            double start = static_cast<double>(cv::getTickCount());
            auto outNames1 = new Tensor(model, opers[opers.size() - 1]);
            model.run(inpName, {outNames1});
            std::vector<float> data = outNames1->get_data<float>();
            int nclass = 12;
            float max = 0.0;
            std::vector<int> chars;
            std::vector<double> scores;
            double time = ((double)cv::getTickCount() - start) / cv::getTickFrequency();
            for (auto index : outNames1->dim())
            {
                std::cout << index << std::endl;
            }
            for (int i = 0; i < 16; i++)
            {
                int index = 12 - 1;
                for (int j = 0; j < 12; j++)
                {
                    if (data[i * nclass + j] > max)
                    {
                        max = data[i * nclass + j];
                        index = j;
                    }
                }
                chars.push_back(index);
                scores.push_back(max);
                max = 0.0;
            }

            string add_str;
            while (ifs)
            {
                getline(ifs, add_str);
                labels_.push_back(add_str);
            }
            ifs.close();
            int x = 0;
            std::string value = "";
            for (int i = 0; i < chars.size(); i++)
            {

                if (chars[i] != nclass - 1 &&
                    ((!(i > 0 && chars[i] == chars[i - 1])) || (i > 1 && chars[i] == chars[i - 2])))
                {
                    std::string value1 = labels_[chars[i] + 1];
                    if (scores[i] < 0.5)
                    {

                        continue;
                    }

                    value += value1;
                }
            }
            std::cout << value << std::endl;
            std::cout << "time : " << time << std::endl;

            auto dims = outNames1->dim();
            for (auto value : dims)
            {
                std::cout << value << std::endl;
            }
        });
        th.detach();
    }

    cv::imshow("Image", img);
    cv::waitKey(0);*/
}
