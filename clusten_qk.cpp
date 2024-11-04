/******************************************************************************
###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################
*******************************************************************************/

#include "hpu_custom_op.h"
#include <torch/extension.h>
#include <perf_lib_layer_params.h>

bool register_clusten_qk() {
    // Registering custom_op::custom_add
    // inputs desc
    habana::custom_op::InputDesc input_a_desc{
        habana::custom_op::input_type::TENSOR, 0};
    habana::custom_op::InputDesc input_b_desc{
        habana::custom_op::input_type::TENSOR, 1};
    habana::custom_op::InputDesc input_c_desc{
        habana::custom_op::input_type::TENSOR, 2};

    std::vector<habana::custom_op::InputDesc> inputs_desc{
        input_a_desc, input_b_desc, input_c_desc};

    // output desc
    // output shape callback
    auto output_size_lambda =
        [](const at::Stack& inputs) -> std::vector<int64_t> {
      auto query = inputs[0].toTensor(); // input
      auto nbhd_idx = inputs[2].toTensor(); // input
      std::vector<int64_t> result_sizes = query.sizes().vec();
      result_sizes[3] = nbhd_idx.sizes()[2];
      return result_sizes;
    };

    habana::custom_op::OutputDesc output_desc{
        0, c10::ScalarType::Float, output_size_lambda};

    std::vector<habana::custom_op::OutputDesc> outputs_desc{
        output_desc};

    // actual register
    REGISTER_CUSTOM_OP_ATTRIBUTES(
        "custom_op::clusten_qk", //schema name
        "clusten_qk_fwd_f32_gaudi2", // guid
        inputs_desc,
        outputs_desc,
        nullptr);
    std::cout << "cpp registered custom_op::clusten_qk\n";
    return true;
}

bool register_custom_relu_backward() {
    // Registering custom_op::custom_add
    // inputs desc
    habana::custom_op::InputDesc input_a_desc{
        habana::custom_op::input_type::TENSOR, 0};
    habana::custom_op::InputDesc input_b_desc{
        habana::custom_op::input_type::TENSOR, 1};
    habana::custom_op::InputDesc input_c_desc{
        habana::custom_op::input_type::TENSOR, 2};
    habana::custom_op::InputDesc input_d_desc{
        habana::custom_op::input_type::TENSOR, 3};

    std::vector<habana::custom_op::InputDesc> inputs_desc{
        input_a_desc, input_b_desc, input_c_desc, input_d_desc};

    // output desc
    // output shape callback
    auto output_size_lambda =
        [](const at::Stack& inputs) -> std::vector<int64_t> {
      auto self = inputs[0].toTensor(); // input
      std::vector<int64_t> result_sizes = self.sizes().vec();
      return result_sizes;
    };

    habana::custom_op::OutputDesc output_desc{
        0, c10::ScalarType::Float, output_size_lambda};

    std::vector<habana::custom_op::OutputDesc> outputs_desc{
        output_desc};

    // actual register
    REGISTER_CUSTOM_OP_ATTRIBUTES(
        "custom_op::clusten_qk_backward", //schema name
        "clusten_qk_bwd_f32_gaudi2", // guid
        inputs_desc,
        outputs_desc,
        nullptr);
    std::cout << "cpp registered custom_op::clusten_qk_backward\n";
    return true;
}

at::Tensor clusten_qk_execute(
    torch::Tensor query, torch::Tensor key, torch::Tensor nbhd_idx) {
  TORCH_CHECK(query.scalar_type() == c10::ScalarType::Float, "Input query expected to be Float tensor");
  TORCH_CHECK(key.scalar_type() == c10::ScalarType::Float, "Input key expected to be Float tensor");
  TORCH_CHECK(nbhd_idx.scalar_type() == c10::ScalarType::Long, "Input nbhd_idx expected to be Long tensor");
  // Registering the custom op, need to be called only once
  static bool registered = register_clusten_qk();
  TORCH_CHECK(registered, "clusten qk kernel not registered");
  std::vector<c10::IValue> inputs{input_a};
  // Get custom op descriptor from registry
  auto op_desc = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("custom_op::clusten_qk");
  // Actual call for op execution
  std::vector<at::Tensor> output = op_desc.execute(inputs);
  // op_desc.execute will always return a vector
  return output[0];
}

at::Tensor clusten_qk_backward_execute(
    torch::Tensor input_a,
    torch::Tensor input_b,
    c10::Scalar threshold) {
  TORCH_CHECK(input_a.scalar_type() == c10::ScalarType::Float, "Input input_a expected to be Float tensor");
  TORCH_CHECK(input_b.scalar_type() == c10::ScalarType::Float, "Input input_b expected to be Float tensor");
  TORCH_CHECK(threshold.to<float>() == 0.0, "Threshold values other than 0 are not supported")
  // Registering the custom op, need to be called only once
  static bool registered = register_custom_relu_backward();
  TORCH_CHECK(registered, "custom_relu_backward kernel not registered" );
  std::vector<c10::IValue> inputs{input_a, input_b, threshold};
  // Get custom op descriptor from registry
  auto op_desc = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("custom_op::custom_relu_backward");
  // Actual call for op execution
  std::vector<at::Tensor> output = op_desc.execute(inputs);
  // op_desc.execute will always return a vector
  return output[0];
}

TORCH_LIBRARY(custom_op, m) {
  m.def("clusten_qk(Tensor query, Tensor key, Tensor nbhd_idx) -> Tensor");
  m.def("clusten_qk_backward(Tensor grad_attn, Tensor query, Tensor key, Tensor nbhd_idx) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
  m.impl("clusten_qk", clusten_qk_execute);
  m.impl("clusten_qk_backward", clusten_qk_backward_execute);
}

