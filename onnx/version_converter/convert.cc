// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/version_converter/convert.h"

#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include "onnx/common/ir_pb_converter.h"
#include "onnx/shape_inference/implementation.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

// Return the version of the default-domain ("" or "ai.onnx") opset import, or
// nullopt if the imports declare no default-domain dependency.
static std::optional<int64_t> GetDefaultDomainOpsetVersion(
    const google::protobuf::RepeatedPtrField<OperatorSetIdProto>& opset_import) {
  for (const auto& opset : opset_import) {
    if (opset.domain().empty() || opset.domain() == "ai.onnx") {
      return opset.version();
    }
  }
  return std::nullopt;
}

ModelProto ConvertVersion(const ModelProto& mp_in, int target_version) {
  // Get initial_opsetid from mp_in
  OpSetID initial_struct(0);
  if (const std::optional<int64_t> initial_version = GetDefaultDomainOpsetVersion(mp_in.opset_import())) {
    initial_struct.setVersion(*initial_version);
  }
  OpSetID target_struct = OpSetID(target_version);
  DefaultVersionConverter v;
  return v.convert_version(mp_in, initial_struct, target_struct);
}

// Wrap a function body in a single-graph ModelProto so it can be run through the
// version converter. The function's declared inputs/outputs carry no type of
// their own; supply types from the function's value_info when available.
static ModelProto FunctionBodyAsModel(const FunctionProto& fp, int64_t ir_version) {
  ModelProto model;
  model.set_ir_version(ir_version);
  *model.mutable_opset_import() = fp.opset_import();
  GraphProto* graph = model.mutable_graph();
  graph->set_name(fp.name());

  std::unordered_map<std::string, const TypeProto*> type_of;
  for (const auto& vi : fp.value_info()) {
    if (vi.has_type()) {
      type_of[vi.name()] = &vi.type();
    }
  }
  for (const auto& name : fp.input()) {
    ValueInfoProto* vip = graph->add_input();
    vip->set_name(name);
    const auto it = type_of.find(name);
    if (it != type_of.end()) {
      *vip->mutable_type() = *it->second;
    }
  }
  for (const auto& name : fp.output()) {
    ValueInfoProto* vip = graph->add_output();
    vip->set_name(name);
    const auto it = type_of.find(name);
    if (it != type_of.end()) {
      *vip->mutable_type() = *it->second;
    }
  }
  *graph->mutable_value_info() = fp.value_info();
  *graph->mutable_node() = fp.node();
  return model;
}

// Build a Constant node that outputs the given (named) tensor. Used to relocate a
// graph initializer into a function body, which cannot hold initializers.
static NodeProto MakeConstantNode(const TensorProto& tensor) {
  ONNX_ASSERTM(!tensor.name().empty(), "Cannot build a Constant node for an unnamed tensor")
  NodeProto constant;
  constant.set_op_type("Constant");
  constant.set_domain("");
  constant.add_output(tensor.name());
  AttributeProto* value = constant.add_attribute();
  value->set_name("value");
  value->set_type(AttributeProto_AttributeType_TENSOR);
  *value->mutable_t() = tensor;
  value->mutable_t()->clear_name();
  return constant;
}

// Rank of the tensor type declared on a graph input/output, or -1 if it
// carries no tensor shape.
static int TensorTypeRank(const ValueInfoProto& value_info) {
  if (!value_info.has_type() || !value_info.type().has_tensor_type() || !value_info.type().tensor_type().has_shape()) {
    return -1;
  }
  return value_info.type().tensor_type().shape().dim_size();
}

// Adapters keep a conversion local to their node (e.g. Scan_8_9 squeezes the
// batch axis away around the converted Scan instead of re-ranking the values
// surrounding it), so converting a function body must leave the function's
// formal interface -- and therefore its call sites -- unchanged. Verify that,
// refusing loudly if an adapter breaks the invariant, rather than emit a
// function that no longer matches its callers.
static void AssertInterfacePreserved(
    const std::string& function_name,
    int target_version,
    const google::protobuf::RepeatedPtrField<ValueInfoProto>& before,
    const google::protobuf::RepeatedPtrField<ValueInfoProto>& after) {
  ONNX_ASSERTM(
      before.size() == after.size(),
      "Cannot convert function '",
      function_name,
      "' to opset ",
      target_version,
      ": conversion changed the number of formal inputs/outputs")
  for (int i = 0; i < before.size(); ++i) {
    const int rank_before = TensorTypeRank(before.Get(i));
    const int rank_after = TensorTypeRank(after.Get(i));
    ONNX_ASSERTM(
        rank_before < 0 || rank_after < 0 || rank_before == rank_after,
        "Cannot convert function '",
        function_name,
        "' to opset ",
        target_version,
        ": conversion would change the rank of '",
        before.Get(i).name(),
        "' from ",
        rank_before,
        " to ",
        rank_after,
        ", which cannot be expressed on the function's interface")
  }
}

// Convert a local function's body to the target default-domain opset.
static FunctionProto ConvertFunctionVersion(const FunctionProto& fp_in, int target_version, int64_t ir_version) {
  const std::optional<int64_t> initial_version = GetDefaultDomainOpsetVersion(fp_in.opset_import());
  if (!initial_version || *initial_version == target_version) {
    return fp_in;
  }

  ModelProto wrapper = FunctionBodyAsModel(fp_in, ir_version);

  // Some adapters (e.g. the broadcast 6->7) require operand shapes to be
  // present. For the main graph, the version converter relies on the shape
  // inference pass that the Python entry point runs, but that pass does
  // not descend into function bodies, so it's done here.
  shape_inference::InferShapes(wrapper);

  ModelProto converted = ConvertVersion(wrapper, target_version);
  AssertInterfacePreserved(fp_in.name(), target_version, wrapper.graph().input(), converted.graph().input());
  AssertInterfacePreserved(fp_in.name(), target_version, wrapper.graph().output(), converted.graph().output());

  // Rebuild the function from the converted ModelProto.
  FunctionProto fp_out = fp_in;
  fp_out.clear_opset_import();
  *fp_out.mutable_opset_import() = converted.opset_import();
  fp_out.clear_node();

  // A FunctionProto has no initializer field. Any initializer here was created
  // by an adapter (e.g. Pad 10->11 moving the pads from an attribute to an
  // input, creating an initializer for the pads). This loop converts each
  // into a Constant node.
  for (const auto& initializer : converted.graph().initializer()) {
    *fp_out.add_node() = MakeConstantNode(initializer);
  }
  // No adapters currently produce a sparse initializer, but in the future they
  // could also be converted the same way.
  ONNX_ASSERTM(
      converted.graph().sparse_initializer().empty(),
      "Unsupported: sparse initializer produced while converting function body")

  // Append the converted body after any Constants
  fp_out.mutable_node()->MergeFrom(converted.graph().node());

  return fp_out;
}

void DefaultVersionConverter::convert_graph(
    const std::shared_ptr<Graph>& g,
    const OpSetID& initial_version,
    const OpSetID& target_version) const {
  assertNonNull(g);

  // TODO(ONNX): Move to Inter-Domain Converter
  // Get initial model versions
  // std::vector<OpSetID> initial_versions = g->opset_versions_mutable();

  // No conversion necessary if Model has single, equivalent opset version
  // if (initial_versions.size() == 1 && initial_versions[0].version ==
  //    target_version.version && initial_versions[0].domain ==
  //    target_version.domain) {
  //  return mp_in;
  // }

  // Check if versions are valid
  assertInVersionRange(initial_version.version());
  assertInVersionRange(target_version.version());

  // Iterate over all versions to target_version for specified
  int64_t curr_version = initial_version.version();
  int64_t step = 0;
  if (target_version.version() > initial_version.version()) {
    step = 1;
  } else {
    step = -1;
  }
  // Identify index of the default domain ("" or "ai.onnx") in g.opset_versions.
  // ImportModelProto preserves domain strings verbatim from the proto, so both
  // spellings must be matched here (ConvertVersion also accepts both).
  int domain_index = -1;
  for (int i = 0; i < static_cast<int>(g->opset_versions_mutable().size()); i++) {
    const std::string& dom = g->opset_versions_mutable()[i].domain();
    if (dom.empty() || dom == "ai.onnx") {
      domain_index = i;
      break;
    }
  }
  ONNX_ASSERTM(domain_index >= 0, "Graph has no default-domain (\"\" or \"ai.onnx\") opset entry")
  while (curr_version != target_version.version()) {
    debug(
        "curr_version: " + ONNX_NAMESPACE::to_string(curr_version) +
        ", next_version: " + ONNX_NAMESPACE::to_string(curr_version + step));
    Node* cur_op = nullptr;
    graph_node_list_iterator it = g->begin();
    // Iterate through and call adapter returned by adapter_lookup for ops from
    // current_version opset. We have to manipulate the iterator explicitly because cur_op
    // might change when applying the adapter (e.g. for deprecated ops)
    while (it != g->end()) {
      cur_op = *it;
      debug(std::string("Finding schema for ") + std::string(cur_op->kind().toString()));
      const std::string op_name = cur_op->kind().toString();
      if (op_name == "ConstantFill") {
        if (DEBUG) {
          std::cerr
              << "Warning: skipping schema search for experimental op 'ConstantFill' and keeping the op as is. "
                 "Please be advised the converted model may not be working properly if target runtime does not support this "
                 "experimental op."
              << '\n';
        }
      } else if (!cur_op->domain().empty() && cur_op->domain() != "ai.onnx") {
        if (DEBUG) {
          std::cerr << "Warning: opset domain '" << cur_op->domain() << "' is not supported." << '\n';
        }
      } else if (op_name != "Undefined" && op_name != "Captured") {
        const auto schema_it = all_schemas.find(op_name);
        ONNX_ASSERTM(
            schema_it != all_schemas.end(),
            "Op '%s' has no registered schema; cannot convert it from version %lld to %lld.",
            op_name.c_str(),
            static_cast<long long>(curr_version),
            static_cast<long long>(target_version.version()));
        const auto& op_domain_map = schema_it->second;
        OpSetID curr_id(curr_version);
        OpSetID next_id(curr_version + step);
        if (searchOpDomainMap(op_domain_map, curr_version, step)) {
          // Op is specifically defined for this domain and version
          const auto& op_adapter = adapter_lookup(cur_op, curr_id, next_id);
          // If adapter_lookup returns null, no adapter is present.
          // Error thrown by adapter_lookup
          if (DEBUG) {
            std::cerr << "Applying adapter" << '\n';
          }
          // adapt should handle replacing node in graph
          cur_op = op_adapter.adapt(g, cur_op);
          it = graph_node_list_iterator(cur_op, kNextDirection);
        }
        // Recursively convert any subgraph attributes
        for (const auto& attr : cur_op->attributeNames()) {
          if (cur_op->kindOf(attr) == AttributeKind::g) {
            convert_graph(cur_op->g(attr), curr_id, next_id);
          }
        }
      }
      ++it;
    }
    // Update model version
    curr_version += step;
    g->opset_versions_mutable()[static_cast<size_t>(domain_index)].incrementVersion(step);
  }
}

ModelProto DefaultVersionConverter::convert_version(
    const ModelProto& mp_in,
    const OpSetID& initial_version,
    const OpSetID& target_version) const {
  const std::string& initial_domain = initial_version.domain();
  const std::string& target_domain = target_version.domain();
  assertDefaultDomain(initial_domain, target_domain);

  for (auto it = mp_in.opset_import().begin(); it != mp_in.opset_import().end(); ++it) {
    if (it->domain() == initial_version.domain()) {
      ONNX_ASSERTM(
          initial_version.version() == it->version(), "initial_version does not reflect current state of model")
    }
  }

  std::shared_ptr<Graph> g(ImportModelProto(mp_in));

  convert_graph(g, initial_version, target_version);

  // Export g as ModelProto
  debug("Finished conversion; returning model");
  ModelProto mp_out = PrepareOutput(mp_in);
  ExportModelProto(&mp_out, g);

  // Convert local functions.
  for (const auto& fp_in : mp_in.functions()) {
    *mp_out.add_functions() =
        ConvertFunctionVersion(fp_in, static_cast<int>(target_version.version()), mp_in.ir_version());
  }

  return mp_out;
}

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
