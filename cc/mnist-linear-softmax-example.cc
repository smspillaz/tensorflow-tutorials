/*
 * /cc/mnist-linear-softmax-example.cc
 *
 * An example of parsing the MNIST data and learning
 * a simple linear model with single nonlinearity
 * at the end using the TensorFlow C++ API.
 *
 * See /LICENSE.md for Copyright information
 */
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

#include <algorithm>
#include <fstream>
#include <iosfwd>
#include <string>
#include <sstream>

namespace tf = ::tensorflow;
using namespace tensorflow::ops;

struct Options
{
    std::string data_dir;
};

Options parse_options(int argc, char **argv) {
    if (argc != 2) {
        LOG(ERROR) << "Must provide a single argument, data_dir";
        exit(1);
    }

    Options options;
    options.data_dir = std::string(argv[1]);

    return options;
}

struct MNISTFiles {
    MNISTFiles(std::ifstream &&train_images,
               std::ifstream &&train_labels,
               std::ifstream &&test_images,
               std::ifstream &&test_labels) :
        train_images(std::move(train_images)),
        train_labels(std::move(train_labels)),
        test_images(std::move(test_images)),
        test_labels(std::move(test_labels))
    {
    }

    MNISTFiles(MNISTFiles const &) = default;
    MNISTFiles(MNISTFiles &&) = default;
    MNISTFiles & operator=(MNISTFiles const &) = default;
    MNISTFiles & operator=(MNISTFiles &&) = default;

    std::ifstream train_images;
    std::ifstream train_labels;

    std::ifstream test_images;
    std::ifstream test_labels;
    
};

namespace detail {
    template<typename String>
    void join_strings(std::stringstream &stream, String &&str)
    {
        stream << str;
    }

    template<typename String, typename... Strings>
    void join_strings(std::stringstream &stream, String &&str, Strings&&... strs)
    {
        stream << str;
        join_strings(stream, std::forward<Strings>(strs)...);
    }
}

template<typename... Strings>
std::string join_strings(Strings&&... strs) {
    std::stringstream ss;
    detail::join_strings(ss, std::forward<Strings>(strs)...);
    return ss.str();
}

MNISTFiles find_files_in_directory(std::string const &data_dir) {
    return MNISTFiles(std::ifstream(join_strings(data_dir,
                                                 "/",
                                                 "train-images-idx3-ubyte"),
                                    std::ifstream::in | std::ifstream::binary),
                      std::ifstream(join_strings(data_dir,
                                                 "/",
                                                 "train-labels-idx1-ubyte"),
                                    std::ifstream::in | std::ifstream::binary),
                      std::ifstream(join_strings(data_dir,
                                                 "/",
                                                 "t10k-images-idx3-ubyte"),
                                    std::ifstream::in | std::ifstream::binary),
                      std::ifstream(join_strings(data_dir,
                                                 "/",
                                                 "t10k-labels-idx1-ubyte"),
                                    std::ifstream::in | std::ifstream::binary));
}

std::vector<float_t>
one_hot_encode (size_t index, size_t size)
{
    std::vector<float_t> vec(size);
    std::fill(vec.begin(), vec.end(), 0.0f);

    vec[index] = 1.0f;
    return vec;
}

int32_t read_int32_byteswap(std::ifstream &stream) {
    int32_t location;
    stream.read(reinterpret_cast<char *>(&location), sizeof(int32_t));

    unsigned char *bytes = reinterpret_cast<unsigned char *>(&location);

    int32_t result = (bytes[0] << 24);
    result |= (bytes[1] << 16);
    result |= (bytes[2] << 8);
    result |= bytes[3];

    return result;
}

std::vector<tf::Tensor>
parse_idx1_one_hot(std::ifstream &stream) {
    static const uint32_t EXPECTED_MAGIC = 0x00000801;
    std::vector<tf::Tensor> idx1_tensors;

    stream.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    uint32_t magic = read_int32_byteswap(stream);

    if (magic != EXPECTED_MAGIC) {
        LOG(ERROR) << "Expected magic to be " << EXPECTED_MAGIC << " got " << magic;
        exit(1);
    }

    int32_t n_items = read_int32_byteswap(stream);

    while (n_items--) {
        char label;

        stream.get(label);
        auto encoded_label = one_hot_encode(label, 10);
  
        idx1_tensors.emplace_back(tf::Tensor(tf::DataTypeToEnum<float>::v(),
                                             tf::TensorShape({10})));
        copy_n(encoded_label.begin(),
               encoded_label.size(),
               idx1_tensors.back().flat<float>().data());
    }

    return idx1_tensors;
}

std::vector<tf::Tensor>
parse_idx3(std::ifstream &stream) {
    static const uint32_t EXPECTED_MAGIC = 0x00000803;
    std::vector<tf::Tensor> idx3_tensors;

    stream.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    uint32_t magic = read_int32_byteswap(stream);

    if (magic != EXPECTED_MAGIC) {
        LOG(ERROR) << "Expected magic to be " << EXPECTED_MAGIC << " got " << magic;
        exit(1);
    }

    int32_t n_items = read_int32_byteswap(stream);
    int32_t rows = read_int32_byteswap(stream);
    int32_t columns = read_int32_byteswap(stream);
    int32_t pixels = rows * columns;

    for (size_t i = 0; i < n_items; ++i) {
        std::vector<float_t> normalized_pixel_data(pixels);
        for (size_t j = 0; j < pixels; ++j) {
            char pixel;

            stream.get(pixel);
            normalized_pixel_data[j] = static_cast<float_t>(pixel) / 255.0f;
        }
  
        idx3_tensors.emplace_back(tf::Tensor(tf::DataTypeToEnum<float>::v(),
                                             tf::TensorShape({pixels})));
        copy_n(normalized_pixel_data.begin(),
               normalized_pixel_data.size(),
               idx3_tensors.back().flat<float>().data());
    }

    return idx3_tensors;
}

struct MNISTData {
    MNISTData(std::vector<tf::Tensor> &&train_data,
              std::vector<tf::Tensor> &&train_labels,
              std::vector<tf::Tensor> &&test_data,
              std::vector<tf::Tensor> &&test_labels):
        train_data(std::move(train_data)),
        train_labels(std::move(train_labels)),
        test_data(std::move(test_data)),
        test_labels(std::move(test_labels))
    {
    }

    MNISTData(MNISTData const &) = default;
    MNISTData(MNISTData &&) = default;
    MNISTData & operator=(MNISTData const &) = default;
    MNISTData & operator=(MNISTData &&) = default;

    std::vector<tf::Tensor> train_data;
    std::vector<tf::Tensor> train_labels;

    std::vector<tf::Tensor> test_data;
    std::vector<tf::Tensor> test_labels;
};

MNISTData load_data_from_files(MNISTFiles &files) {
    return MNISTData(parse_idx3(files.train_images),
                     parse_idx1_one_hot(files.train_labels),
                     parse_idx3(files.test_images),
                     parse_idx1_one_hot(files.test_labels));
}

int main(int argc, char **argv) {
    Options options(parse_options(argc, argv));
    MNISTFiles files(find_files_in_directory(options.data_dir));
    LOG(INFO) << "Opened MNIST files";

    MNISTData loaded_data(load_data_from_files(files));
    LOG(INFO) << "Parsed MNIST data";

    return 0;
}