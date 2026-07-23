// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <string>

#include "gtest/gtest.h"
#include "onnx/common/ir.h"
#include "onnx/common/ir_pb_converter.h"
#include "onnx/defs/parser.h"

namespace ONNX_NAMESPACE {
namespace Test {

static bool IsValidIdentifier(const std::string& name) {
  if (name.empty()) {
    return false;
  }
  if (!IsAlpha(name[0]) && name[0] != '_') {
    return false;
  }
  for (size_t i = 1; i < name.size(); ++i) {
    if (!IsAlnum(name[i]) && name[i] != '_') {
      return false;
    }
  }
  return true;
}

TEST(IR, ValidIdentifierTest) {
  Graph* g = new Graph(); // NOLINT(cppcoreguidelines-owning-memory)
  g->setName("test");
  Value* x = g->addInput();
  x->setUniqueName("x");
  x->setElemType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  x->setSizes({Dimension("M"), Dimension("N")});
  Node* node1 = g->create(kNeg, 1);
  node1->addInput(x);
  g->appendNode(node1);
  Value* temp1 = node1->outputs()[0];
  Node* node2 = g->create(kNeg, 1);
  node2->addInput(temp1);
  g->appendNode(node2);
  Value* y = node2->outputs()[0];
  g->registerOutput(y);

  ModelProto model;
  ExportModelProto(&model, std::shared_ptr<Graph>(g));

  for (const auto& node : model.graph().node()) {
    for (const auto& name : node.output()) {
      EXPECT_TRUE(IsValidIdentifier(name));
    }
  }
}

// Regression test: Tensor::elem_num() and size_from_dim() must use 64-bit
// arithmetic. Previously, std::accumulate used `1` (int) as the initial value,
// causing 32-bit multiplication that silently overflowed for tensors whose
// element count exceeded INT_MAX (~2.1B). Fixed by using int64_t{1}.
TEST(Tensor, ElemNumLargeTensorNoOverflow) {
  Tensor t;
  // 50000 * 50000 = 2,500,000,000 which exceeds INT32_MAX (2,147,483,647)
  t.sizes() = {50000, 50000};
  const int64_t expected = static_cast<int64_t>(50000) * 50000;
  EXPECT_EQ(t.elem_num(), expected);
  EXPECT_EQ(t.size_from_dim(0), expected);
  EXPECT_EQ(t.size_from_dim(1), int64_t{50000});
}

// The AttributeProto.ref_attr_name field (for an attribute with no value that
// is a reference to a function attribute) must survive the ImportModelProto ->
// ExportModelProto round-trip.
TEST(IR, ReferenceAttributeRoundTrip) {
  ModelProto model_in;
  model_in.set_ir_version(IR_VERSION);
  auto* opset = model_in.add_opset_import();
  opset->set_domain("");
  opset->set_version(16);
  auto* graph = model_in.mutable_graph();
  graph->set_name("g");

  auto* input = graph->add_input();
  input->set_name("x");
  input->mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  auto* output = graph->add_output();
  output->set_name("y");
  output->mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);

  auto* node = graph->add_node();
  node->set_op_type("LeakyRelu");
  node->add_input("x");
  node->add_output("y");
  // A reference attribute: names the referenced parent-function attribute and
  // a type, but carries no value.
  // This is technically invalid to set outside a FunctionProto: but since the
  // IR doesn't model-local functions this is the only way to set up a model
  // that covers this path.
  auto* attr = node->add_attribute();
  attr->set_name("alpha");
  attr->set_type(AttributeProto_AttributeType_FLOAT);
  attr->set_ref_attr_name("alpha");

  // Import: the reference must be represented in the IR, not dropped.
  std::shared_ptr<Graph> g = ImportModelProto(model_in);
  ASSERT_TRUE(g != nullptr);
  Node* imported = nullptr;
  for (Node* n : g->nodes()) {
    if (n->kind() == Symbol("LeakyRelu")) {
      imported = n;
      break;
    }
  }
  ASSERT_NE(imported, nullptr);
  const Symbol alpha("alpha");
  ASSERT_TRUE(imported->hasAttribute(alpha));
  EXPECT_TRUE(imported->isReference(alpha));
  EXPECT_EQ(imported->kindOf(alpha), AttributeKind::ref);
  EXPECT_EQ(imported->refAttrName(alpha), "alpha");
  EXPECT_EQ(imported->refType(alpha), AttributeProto_AttributeType_FLOAT);

  // Export: ref_attr_name and type survive, with no value field set.
  ModelProto model_out;
  ExportModelProto(&model_out, g);
  ASSERT_EQ(model_out.graph().node_size(), 1);
  const auto& out_node = model_out.graph().node(0);
  ASSERT_EQ(out_node.attribute_size(), 1);
  const auto& out_attr = out_node.attribute(0);
  EXPECT_EQ(out_attr.name(), "alpha");
  EXPECT_EQ(out_attr.ref_attr_name(), "alpha");
  EXPECT_EQ(out_attr.type(), AttributeProto_AttributeType_FLOAT);
  EXPECT_FALSE(out_attr.has_f());
}

} // namespace Test
} // namespace ONNX_NAMESPACE
