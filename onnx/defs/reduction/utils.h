// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <string>
#include <vector>

#include "onnx/defs/schema.h"
#include "onnx/defs/tensor_proto_util.h"

namespace ONNX_NAMESPACE {

// Constants used to indicate value returned by reduction of an empty set of values.
constexpr const char* EMPTY_ZERO = "0";
constexpr const char* EMPTY_ONE = "1";
constexpr const char* EMPTY_UNDEFINED = "undefined";
constexpr const char* EMPTY_MIN =
    "minus infinity (if supported by the datatype) or the minimum value of the data type otherwise";
constexpr const char* EMPTY_MAX =
    "plus infinity (if supported by the datatype) or the maximum value of the data type otherwise";
constexpr const char* EMPTY_MINUS_INF = "minus infinity (if supported by the datatype) or undefined otherwise";

// Types accepted by ReduceMin and ReduceMax: every numeric type, plus bool.
// These reductions select one of the input elements rather than combining them,
// so no accumulation can overflow and no width needs to be excluded. This is the
// same set ArgMin and ArgMax use, which reduce over the same axes to pick an
// element rather than a value.
const std::vector<std::string>& MinMaxReductionTypes();

// The remaining reductions accumulate over the reduced elements, so they take
// OpSchema::numeric_types_for_math_reduction_ir4(), which omits the 8- and
// 16-bit integers to keep a wide accumulator. Call sites name the set
// explicitly: the old.cc schemas are frozen at their published types, and a
// shared flag would silently move them.
std::function<void(OpSchema&)> ReduceOpGenerator(
    const char* name,
    const char* empty_value,
    const std::vector<std::string>& allowed_types,
    bool axes_input = false,
    const char* func_body = nullptr,
    const ContextDependentFunctionBodyBuilder& function_builder = nullptr);

inline std::function<void(OpSchema&)> ReduceOpDynamicAxes(const char* name, const char* empty_value) {
  return ReduceOpGenerator(name, empty_value, OpSchema::numeric_types_for_math_reduction_ir4(), true);
}

inline std::function<void(OpSchema&)>
ReduceFunctionOp(const char* name, const char* empty_value, const char* func_body) {
  return ReduceOpGenerator(name, empty_value, OpSchema::numeric_types_for_math_reduction_ir4(), true, func_body);
}

} // namespace ONNX_NAMESPACE
