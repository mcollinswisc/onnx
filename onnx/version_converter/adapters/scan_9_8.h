// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Adapter for Scan in default domain from version 9 to 8

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

struct Scan_9_8 final : public Adapter {
  explicit Scan_9_8() : Adapter("Scan", OpSetID(9), OpSetID(8)) {}

  void adapt_scan_9_8(const std::shared_ptr<Graph>& graph, Node* node) const {
    ONNX_ASSERTM(node != nullptr, "Scan node is null")
    ONNX_ASSERTM(node->owningGraph() != nullptr, "Scan node does not belong to a graph")
    const std::vector<Value*> inputs(node->inputs().vec());
    const std::vector<Value*> outputs(node->outputs().vec());

    // Handling Attribute Changes

    Symbol input_dirs = Symbol("scan_input_directions");
    if (node->hasAttribute(input_dirs)) {
      std::vector<int64_t> scan_input_directions(node->is(input_dirs));
      node->removeAttribute(input_dirs);
      node->is_(Symbol("directions"), std::move(scan_input_directions));
    }

    Symbol output_dirs = Symbol("scan_output_directions");
    if (node->hasAttribute(output_dirs)) {
      const std::vector<int64_t> scan_output_directions(node->is(output_dirs));
      for (int64_t x : scan_output_directions) {
        ONNX_ASSERTM(x == 0, "Unsupported output direction for Version 8")
      }
      node->removeAttribute(output_dirs);
    }

    Symbol input_axes = Symbol("scan_input_axes");
    if (node->hasAttribute(input_axes)) {
      const std::vector<int64_t> scan_input_axes(node->is(input_axes));
      for (int64_t x : scan_input_axes) {
        ONNX_ASSERTM(x == 0, "Unsupported input axes for Version 8")
      }
      node->removeAttribute(input_axes);
    }

    Symbol output_axes = Symbol("scan_output_axes");
    if (node->hasAttribute(output_axes)) {
      const std::vector<int64_t> scan_output_axes(node->is(output_axes));
      for (int64_t x : scan_output_axes) {
        ONNX_ASSERTM(x == 0, "Unsupported output axes for Version 8")
      }
      node->removeAttribute(output_axes);
    }

    // Handling Input and Output Changes
    //
    // Scan in opset 8 carries a leading batch axis that opset 9 removed.
    // Unsqueeze a batch axis of 1 onto each input in front of the Scan and
    // squeeze it away behind, so the values around the node keep their rank
    // and the conversion stays local to the node -- the enclosing graph,
    // subgraph, or function interface is unaffected.

    node->removeAllInputs();

    Value* v = node->owningGraph()->createValue(*node, 0);
    v->setUniqueName("");
    v->setElemType(TensorProto_DataType::TensorProto_DataType_INT32);
    node->addInput(v);

    for (Value* input : inputs) {
      Node* unsqueeze = graph->create(kUnsqueeze);
      unsqueeze->is_(kaxes, std::vector<int64_t>{0});
      unsqueeze->addInput(input);
      unsqueeze->insertBefore(node);
      Value* batched = unsqueeze->output();
      batched->setElemType(input->elemType());
      if (!input->sizes().empty()) {
        std::vector<Dimension> batched_sizes{Dimension(1)};
        batched_sizes.insert(batched_sizes.end(), input->sizes().begin(), input->sizes().end());
        batched->setSizes(batched_sizes);
      }
      node->addInput(batched);
    }

    Node* insert_after = node;
    for (Value* output : outputs) {
      const std::string output_name = output->uniqueName();
      const use_list original_uses(output->uses());
      Node* squeeze = graph->create(kSqueeze);
      squeeze->is_(kaxes, std::vector<int64_t>{0});
      squeeze->insertAfter(insert_after);
      insert_after = squeeze;
      Value* unbatched = squeeze->output();
      unbatched->setElemType(output->elemType());
      if (!output->sizes().empty()) {
        unbatched->setSizes(output->sizes());
        std::vector<Dimension> batched_sizes{Dimension(1)};
        batched_sizes.insert(batched_sizes.end(), output->sizes().begin(), output->sizes().end());
        output->setSizes(batched_sizes);
      }
      output->setUniqueName(output_name + "_intermediate");
      unbatched->setUniqueName(output_name);
      squeeze->addInput(output);
      for (Use u : original_uses) {
        u.user->replaceInputWith(output, unbatched);
      }
    }
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_scan_9_8(graph, node);
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
