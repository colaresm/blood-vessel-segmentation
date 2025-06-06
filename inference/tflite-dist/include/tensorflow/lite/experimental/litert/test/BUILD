# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        # copybara:uncomment "//third_party/mediapipe/calculators/tensor:__subpackages__",
        "//tensorflow/lite/experimental/litert:__subpackages__",
    ],
)

# TODO: b/365295276 - Make custom rule and move to `.sh`.
OUT_DIR = "$(RULEDIR)"

CONVERTER = "//tensorflow/compiler/mlir/lite:tf_tfl_translate"

CMD = """
for mlir_file in $(SRCS); do
    $(location {converter}) --input-mlir $$mlir_file --o={out_dir}/testdata/$$(basename $$mlir_file .mlir).tflite
done
""".format(
    converter = CONVERTER,
    out_dir = OUT_DIR,
)

genrule(
    name = "mlir_test_data",
    srcs = glob(["testdata/*.mlir"]),
    outs = [s.removesuffix(".mlir") + ".tflite" for s in glob(["testdata/*.mlir"])],
    cmd = CMD,
    tools = [CONVERTER],
)

filegroup(
    name = "tflite_test_data",
    srcs = glob(["testdata/*.tflite"]),
)

cc_library(
    name = "common",
    testonly = 1,
    srcs = [
        "common.cc",
    ],
    hdrs = [
        "common.h",
    ],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:c_api_opaque",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core:cc_api_stable",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_model_predicates",
        "//tensorflow/lite/experimental/litert/core:filesystem",
        "//tensorflow/lite/experimental/litert/core/model:model_buffer",
        "//tensorflow/lite/experimental/litert/core/util:flatbuffer_tools",
        "//tensorflow/lite/kernels:builtin_ops",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform",
    ],
)

cc_library(
    name = "simple_model",
    testonly = 1,
    hdrs = [
        "testdata/simple_model_test_vectors.h",
    ],
    data = [
        "testdata/simple_model.tflite",
    ],
    deps = [
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_layout",
    ],
)

cc_library(
    name = "simple_model_npu",
    testonly = 1,
    srcs = [],
    hdrs = [
        "testdata/simple_model_test_vectors.h",
    ],
    data = [
        "testdata/simple_model_google_tensor.bin",
        "testdata/simple_model_mtk.bin",
        "testdata/simple_model_npu.tflite",
        "testdata/simple_model_qualcomm.bin",
    ],
    deps = [
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_layout",
    ],
)

cc_library(
    name = "test_models",
    hdrs = ["test_models.h"],
    deps = [
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "matchers",
    testonly = True,
    hdrs = ["matchers.h"],
    deps = [
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "matchers_test",
    srcs = ["matchers_test.cc"],
    deps = [
        ":matchers",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "@com_google_googletest//:gtest_main",
    ],
)

# Use this library if you want to enforce an OSS environment for your test.
cc_library(
    name = "matchers_oss",
    testonly = True,
    hdrs = ["matchers.h"],
    tags = ["avoid_dep"],
    deps = [
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
    ],
)
