load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

tf_cc_binary(
    name = "tensorflow-tutorials-matmul",
    srcs = [
        "cc/matmul-example.cc"
    ],
    deps = [
        "//tensorflow/cc:gradients",
        "//tensorflow/cc:grad_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:tensorflow"
    ]
)