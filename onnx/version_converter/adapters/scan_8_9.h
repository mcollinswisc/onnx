// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Adapter for Scan in default domain from version 8 to 9

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {
struct Scan_8_9 final : public Adapter {
  explicit Scan_8_9() : Adapter("Scan", OpSetID(8), OpSetID(9)) {}

  void adapt_scan_8_9(const std::shared_ptr<Graph>& graph, Node* node) const {
    const std::vector<Value*> inputs(node->inputs().vec());
    const std::vector<Value*> outputs(node->outputs().vec());

    // Validate preconditions before mutating the node, so a failed assertion
    // doesn't leave the graph in a partially-converted state.
    ONNX_ASSERTM(!inputs.empty(), "Scan node in opset 8 must have at least 1 input")
    ONNX_ASSERTM(inputs[0]->uniqueName().empty(), "Unsupported conversion to opset 9")
    for (size_t i = 1; i < inputs.size(); ++i) {
      const std::vector<Dimension>& sizes = inputs[i]->sizes();
      ONNX_ASSERTM(
          sizes.empty() || !sizes[0].is_int || sizes[0].dim == 1,
          "Unsupported conversion to opset 9: Scan input '",
          inputs[i]->uniqueName(),
          "' has batch_size ",
          sizes[0].dim,
          ", but only batch_size 1 can be expressed in opset 9")
    }

    // Handling Attribute Changes

    Symbol dirs = Symbol("directions");
    if (node->hasAttribute(dirs)) {
      std::vector<int64_t> directions(node->is(dirs));
      node->removeAttribute(dirs);
      node->is_(Symbol("scan_input_directions"), std::move(directions));
    }

    // Handling Input and Output Changes
    //
    // Scan in opset 8 carries a leading batch axis (asserted above to be 1)
    // that opset 9 removed. Squeeze it away in front of the Scan and restore
    // it behind, so the values around the node keep their rank and the
    // conversion stays local to the node -- the enclosing graph, subgraph, or
    // function interface is unaffected.

    node->removeAllInputs();

    // inputs[0] is the optional sequence_lens (asserted empty above); Scan in
    // opset 9 does not have it, so it is intentionally dropped. Every other
    // input is kept -- including one whose shape is unknown, which the Squeeze
    // handles like any other rather than dropping it.
    for (size_t i = 1; i < inputs.size(); ++i) {
      Value* input = inputs[i];
      Node* squeeze = graph->create(kSqueeze);
      squeeze->is_(kaxes, std::vector<int64_t>{0});
      squeeze->addInput(input);
      squeeze->insertBefore(node);
      Value* unbatched = squeeze->output();
      unbatched->setElemType(input->elemType());
      if (!input->sizes().empty()) {
        unbatched->setSizes(std::vector<Dimension>(input->sizes().begin() + 1, input->sizes().end()));
      }
      node->addInput(unbatched);
    }

    Node* insert_after = node;
    for (Value* output : outputs) {
      const std::string output_name = output->uniqueName();
      const use_list original_uses(output->uses());
      Node* unsqueeze = graph->create(kUnsqueeze);
      unsqueeze->is_(kaxes, std::vector<int64_t>{0});
      unsqueeze->insertAfter(insert_after);
      insert_after = unsqueeze;
      Value* batched = unsqueeze->output();
      batched->setElemType(output->elemType());
      if (!output->sizes().empty()) {
        batched->setSizes(output->sizes());
        output->setSizes(std::vector<Dimension>(output->sizes().begin() + 1, output->sizes().end()));
      }
      output->setUniqueName(output_name + "_intermediate");
      batched->setUniqueName(output_name);
      unsqueeze->addInput(output);
      for (Use u : original_uses) {
        u.user->replaceInputWith(output, batched);
      }
    }
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_scan_8_9(graph, node);
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
