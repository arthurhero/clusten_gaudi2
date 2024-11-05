/******************************************************************************
###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################
*******************************************************************************/

#include "hpu_custom_op.h"
#include <torch/extension.h>
#include <perf_lib_layer_params.h>

bool register_clusten_qk_fwd() {
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
        "custom_op::clusten_qk_fwd", //schema name
        "clusten_qk_fwd_f32_gaudi2", // guid
        inputs_desc,
        outputs_desc,
        nullptr);
    std::cout << "cpp registered custom_op::clusten_qk_fwd\n";
    return true;
}

bool register_clusten_qk_bwd() {
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
      auto query = inputs[1].toTensor();
      std::vector<int64_t> result_sizes = query.sizes().vec();
      return result_sizes;
    };

    auto output_size_lambda2 =
        [](const at::Stack& inputs) -> std::vector<int64_t> {
      auto key = inputs[2].toTensor();
      std::vector<int64_t> result_sizes = key.sizes().vec();
      return result_sizes;
    };


    habana::custom_op::OutputDesc output_desc{
        0, c10::ScalarType::Float, output_size_lambda};
    habana::custom_op::OutputDesc output_desc2{
        1, c10::ScalarType::Float, output_size_lambda2};


    std::vector<habana::custom_op::OutputDesc> outputs_desc{
        output_desc, output_desc2};

    // actual register
    REGISTER_CUSTOM_OP_ATTRIBUTES(
        "custom_op::clusten_qk_bwd", //schema name
        "clusten_qk_bwd_f32_gaudi2", // guid
        inputs_desc,
        outputs_desc,
        nullptr);
    std::cout << "cpp registered custom_op::clusten_qk_bwd\n";
    return true;
}

at::Tensor clusten_qk_fwd_execute(
    torch::Tensor query, torch::Tensor key, torch::Tensor nbhd_idx) {
  TORCH_CHECK(query.scalar_type() == c10::ScalarType::Float, "Input query expected to be Float tensor");
  TORCH_CHECK(key.scalar_type() == c10::ScalarType::Float, "Input key expected to be Float tensor");
  TORCH_CHECK(nbhd_idx.scalar_type() == c10::ScalarType::Long, "Input nbhd_idx expected to be Long tensor");
  // Registering the custom op, need to be called only once
  static bool registered = register_clusten_qk_fwd();
  TORCH_CHECK(registered, "clusten qk fwd kernel not registered");
  std::vector<c10::IValue> inputs{query, key, nbhd_idx};
  // Get custom op descriptor from registry
  auto op_desc = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("custom_op::clusten_qk_fwd");
  // Actual call for op execution
  std::vector<at::Tensor> output = op_desc.execute(inputs);
  // op_desc.execute will always return a vector
  return output[0];
}

std::vector<at::Tensor> clusten_qk_bwd_execute(
    torch::Tensor d_attn,
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor nbhd_idx) {
  TORCH_CHECK(d_attn.scalar_type() == c10::ScalarType::Float, "Input d_attn expected to be Float tensor");
  TORCH_CHECK(query.scalar_type() == c10::ScalarType::Float, "Input query expected to be Float tensor");
  TORCH_CHECK(key.scalar_type() == c10::ScalarType::Float, "Input key expected to be Float tensor");
  TORCH_CHECK(nbhd_idx.scalar_type() == c10::ScalarType::Long, "Input nbhd_idx expected to be Long tensor");
  // Registering the custom op, need to be called only once
  static bool registered = register_clusten_qk_bwd();
  TORCH_CHECK(registered, "clusten qk bwd kernel not registered" );
  std::vector<c10::IValue> inputs{d_attn, query, key, nbhd_idx};
  // Get custom op descriptor from registry
  auto op_desc = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("custom_op::clusten_qk_bwd");
  // Actual call for op execution
  std::vector<at::Tensor> output = op_desc.execute(inputs);
  // op_desc.execute will always return a vector
  return output;
}

TORCH_LIBRARY(custom_op, m) {
  m.def("clusten_qk_fwd(Tensor query, Tensor key, Tensor nbhd_idx) -> Tensor");
  m.def("clusten_qk_bwd(Tensor d_attn, Tensor query, Tensor key, Tensor nbhd_idx) -> Tensor[]");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
  m.impl("clusten_qk_fwd", clusten_qk_fwd_execute);
  m.impl("clusten_qk_bwd", clusten_qk_bwd_execute);
}

