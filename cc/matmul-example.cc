/*
 * /cc/matmul-example.cc
 *
 * An example of performing a matrix multiplication on the
 * computation graph using the TensorFlow C++ API.
 *
 * See /LICENSE.md for Copyright information
 */
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

int main() {
    namespace tf = ::tensorflow;
    using namespace tensorflow::ops;

    tf::Scope root(tf::Scope::NewRootScope());
    auto A = Const(root, { { 3.0f, 2.0f }, { -1.0f, 0.0f } });
    auto b = Const(root, { { 3.0f } , { 5.0f } });
    auto v = MatMul(root.WithOpName("v"), A, b);

    std::vector<tf::Tensor> outputs;
    tf::ClientSession session(root);

    TF_CHECK_OK(session.Run({v}, &outputs));
    LOG(INFO) << outputs[0].DebugString();
    return 0;
}