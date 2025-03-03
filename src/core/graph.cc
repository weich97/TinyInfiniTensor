#include "core/graph.h"
#include <algorithm>
#include <numeric>
#include <queue>
#include "operators/transpose.h"
#include "operators/matmul.h"

namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
    //    std::cout << "GraphObj begins: " << std::endl; 
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
    //    std::cout << "GraphObj here0: " << std::endl; 
        int iter = 0;
        for (const auto &tensor : tensors){
    //        std::cout << "iter: " << iter << std::endl;
            iter += 1;
            oss << tensor << "\n";
        }
    //    std::cout << "GraphObj here1: " << std::endl; 
        oss << "Graph operators:\n";
    //    std::cout << "GraphObj here2: " << std::endl;
        iter = 0;
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
    //        std::cout << "GraphObj loop here0: " << op->getOpType().underlying() << std::endl;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
    //        std::cout << "GraphObj loop here1: " << std::endl;
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
    //        std::cout << "GraphObj loop here2: " << std::endl;
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
    //    std::cout << "GraphObj here3: " << std::endl;
    //    oss << "flush" << "\n";
    //    std::cout << "GraphObj here_end: " << std::endl;
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    void GraphObj::deleteConnection(Tensor tensor, Operator op) {
        // if op is target
        IT_ASSERT(std::find(tensor->getTargets().begin(),
                            tensor->getTargets().end(),
                            op) != tensor->getTargets().end());
        tensor->removeTarget(op);
        if (tensor->getSource()) {
            tensor->getSource()->removeSuccessors(op);
            op->removePredecessors(tensor->getSource());
        }
    }
    
    // add op as a target
    void GraphObj::addConnection(Tensor tensor, Operator op) {
        tensor->addTarget(op);
        if (tensor->getSource()) {
            tensor->getSource()->addSuccessors(op);
            op->addPredecessors(tensor->getSource());
        }
    }

    void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        // =================================== 作业 ===================================
        
        // 遍历所有操作符
        int iter = 0;
        for (auto it = ops.begin(); it != ops.end();) {
    //        std::cout << "Here-2: " << ops.size() << std::endl;
            iter += 1;
    //        std::cout << "Here-1: " << iter << std::endl;
            auto op = *it;
    //        std::cout << "Here-0.9: " << iter << std::endl;
            if(!op){
                ++it;
    //            std::cout << "Here-0.8: " << iter << std::endl;
                continue;
            }
    //        std::cout << "Here-0.5: " << iter << std::endl;
            OpType::underlying_t opType = op->getOpType().underlying();
    //        std::cout << "Here0: " << iter << std::endl;
            switch (opType) {
                case OpType::Transpose: {
                    // 获取当前操作符的后继操作符
                    auto nextops = op->getSuccessors();
    //                std::cout << "Here: " << iter << std::endl;
                    if (!nextops.empty() && nextops[0]->getOpType().underlying() == OpType::Transpose) {
                        // 使用 std::dynamic_pointer_cast 进行类型转换
                        auto transposeOp = std::dynamic_pointer_cast<TransposeObj>(op);  // 转换为 shared_ptr<TransposeObj>
                        auto nextTransposeOp = std::dynamic_pointer_cast<TransposeObj>(nextops[0]);
    //                    std::cout << "Here1: " << iter << std::endl;
                        if (transposeOp && nextTransposeOp && transposeOp->getPermute() == nextTransposeOp->getPermute()) {
                            // transpose 只有一个输入
                            auto input = transposeOp->getInputs()[0];
    //                        std::cout << "Here1.5: " << input->getGuid() << std::endl;
                            // 获取第二个 transpose 的输入
                            auto input2 = nextTransposeOp->getInputs()[0];                            

                            // 获取第二个 transpose 的输出
                            auto output = nextTransposeOp->getOutput();
                            
                            removeTensor(input2);
                            removeTensor(output);

                            // 获取第一个 transpose 的前驱操作符
                            auto predecessors_all = transposeOp->getPredecessors();
                            if (!predecessors_all.empty()) {
                                auto predecessor_all = predecessors_all[0];
                                predecessor_all->removeSuccessors(transposeOp);  // 删除前驱与第一个 transpose 的连接
                            }

    //                        std::cout << "Here2: " << iter << std::endl;
                            // 移除当前的 op 连接
                            deleteConnection(input, transposeOp);
    //                        std::cout << "Here3: " << iter << std::endl;

                            // 连接新的操作符（下一个 transpose 操作的后继）
                            auto nextnextop = nextTransposeOp->getSuccessors();
                            if (!nextnextop.empty()) {
                                addConnection(input, nextnextop[0]);  // 连接第一个 transpose 的输入到第二个 transpose 的后继
                                nextnextop[0]->replaceInput(output, input);
                                nextnextop[0]->removePredecessors(nextTransposeOp);  // 删除第二个 transpose 的前驱

                                if (!predecessors_all.empty()) {
                                    // 连接前驱到新的后继
                                    auto predecessor_all = predecessors_all[0];
                                    predecessor_all->addSuccessors(nextnextop[0]);  // 连接前驱到新的后继
                                    nextnextop[0]->addPredecessors(predecessor_all);  // 将前驱作为新的后继的前驱
                                }
                            }
    //                        std::cout << "Here3.5: " << input->getGuid() << std::endl;
    //                        std::cout << "Here4: " << iter << std::endl;
                            // 删除当前和下一个操作符
                            it = ops.erase(it);  // 删除当前的 op，it 会指向下一个元素
                            if (it != ops.end()) {
                                it = ops.erase(it);  // 删除下一个操作符，it 会指向下一个元素
                            }
    //                        std::cout << "Here5: " << iter << std::endl;
    //                        std::cout << "Here6: " << ops.size() << std::endl;
                            continue;  // 跳过当前迭代，避免重复处理已经删除的操作符
                        }
                    }
                    break;
                }
                // 处理其他操作符（例如 Conv, MatMul 等）
                // case OpType::Conv: { ... }
                // case OpType::MatMul: { ... }
                
                case OpType::MatMul: {
                    // 获取当前操作符的输入
                    auto inputs = op->getInputs();
    //                std::cout << "Matmul Here-0.5: " << op->getInputs(0)->getGuid() << std::endl;
                    int freq = 0;

    //                std::cout << "Matmul Here0: " << "freq: " << freq << ", ops.size: " << ops.size() << std::endl;
                    // 遍历所有输入，检查每个输入是否为 Transpose 操作
                    for (auto &input : inputs) {
                        // 获取当前输入的前驱操作符
                        auto predecessor = input->getSource();
    //                    std::cout << "Matmul Here1: " << "freq: " << freq << ", ops.size: " << ops.size() << std::endl;
    //                    std::cout << "Matmul Here1.5: " << "Guid: " << input->getGuid() << ", ops.size: " << ops.size() << std::endl;
                        // 检查前驱是否为空
                        if (predecessor) {
    //                        std::cout << "Matmul Here2: " << "freq: " << freq << ", ops.size: " << ops.size() << std::endl;
                            // 如果前驱是 Transpose 操作
                            if (predecessor->getOpType().underlying() == OpType::Transpose) {
    //                            std::cout << "Matmul Here3: " << "freq: " << freq << ", ops.size: " << ops.size() << std::endl;
                                // 转换为 TransposeObj 类型
                                auto transposeOp = std::dynamic_pointer_cast<TransposeObj>(predecessor);
                                if (transposeOp) {
    //                                std::cout << "Matmul Here4: " << "freq: " << freq << ", ops.size: " << ops.size() << std::endl;
                                    // 如果 transpose 对最后两个维度进行了交换
                                    auto permute = transposeOp->getPermute();
    //                                std::cout << "Matmul Here4.5: " << "freq: " << freq << ", permute.size: " << permute.size() << std::endl;
                                    if (permute[2] == 3 && permute[3] == 2) {
                                        // 获取 Matmul 操作
                                        auto matmulOp = std::dynamic_pointer_cast<MatmulObj>(op);
    //                                    std::cout << "Matmul Here5: " << "freq: " << freq << ", ops.size: " << ops.size() << std::endl;
                                        if (matmulOp) {
                                            if (freq == 0) {
                                                // 第一个输入对应 Transpose 操作，调整 transA
                                                matmulOp->setTransA(!matmulOp->getTransA());
                                            } else if (freq == 1) {
                                                // 第二个输入对应 Transpose 操作，调整 transB
                                                matmulOp->setTransB(!matmulOp->getTransB());
                                            }
    //                                        std::cout << "Matmul Here6: " << "freq: " << freq << ", ops.size: " << ops.size() << std::endl;
                                            // 删除matmul op和其transpose分支上的输入连接
                                            deleteConnection(transposeOp->getOutput(), matmulOp);
    //                                        std::cout << "Matmul Here7: " << "freq: " << freq << ", ops.size: " << ops.size() << std::endl;
                                            // 删除matmul op的transpose分支的输入
                                            removeTensor(input);
    //                                        std::cout << "Matmul Here8: " << "freq: " << freq << ", ops.size: " << ops.size() << std::endl;
                                            // 新建matmul op在原有tranpose分支的输入及连接（不知道要不要插入到原有位置）
                                            auto transpose_input = transposeOp->getInputs()[0];
                                            auto transpose_output = transposeOp->getOutput();
    //                                        std::cout << "Matmul Here9: " << "freq: " << freq << ", ops.size: " << ops.size() << std::endl;
                                            matmulOp->inputs.emplace_back(transpose_input);
    //                                        std::cout << "Matmul Here10: " << "freq: " << freq << ", ops.size: " << ops.size() << std::endl;
                                            transpose_output->removeTarget(matmulOp);
    //                                        std::cout << "Matmul Here11: " << "freq: " << freq << ", ops.size: " << ops.size() << std::endl;
                                            deleteConnection(transpose_input, transposeOp);
    //                                        std::cout << "Matmul Here12: " << "freq: " << freq << ", ops.size: " << ops.size() << std::endl;
                                            addConnection(transpose_input, matmulOp);
    //                                        std::cout << "Matmul Here13: " << "freq: " << freq << ", ops.size: " << ops.size() << std::endl;
                                            //deleteConnection(transpose_output, transposeOp);
                                            //transpose_outputtranspose_output->getSource(transposeOp);
                                            matmulOp->replaceInput(input, transpose_input);
    //                                        std::cout << "Matmul Here14: " << "freq: " << freq << ", ops.size: " << ops.size() << std::endl;
                                            // 更新matmul op的op list

                                            // 如果transposeOp有前驱操作符，需要更新前驱的后继关系
                                            auto predecessors = transposeOp->getPredecessors();
                                            for (auto &predecessor : predecessors) {
                                                // 更新前驱的后继指向matmulOp
                                                matmulOp->addPredecessors(predecessor); 
                                                predecessor->addPredecessors(matmulOp);
                                            } 
                                            
    //                                        std::cout << "Matmul Here15: " << "freq: " << freq << ", ops.size: " << ops.size() << std::endl;

                                            // 删除 transposeOp
                                            auto it_to_remove = std::find(ops.begin(), ops.end(), transposeOp);
                                            it = ops.erase(it_to_remove);
                                            it = std::find(ops.begin(), ops.end(), op);
                                            
    //                                        std::cout << "Matmul Here16: " << "freq: " << freq << ", ops.size: " << ops.size() << std::endl;

                                        }
                                    }
                                }
                            }
                        }
                        freq += 1;
    //                    std::cout << "Matmul Here17: " << "freq: " << freq << ", ops.size: " << ops.size() << std::endl;
                    }
    //                std::cout << "Matmul Here18: " << "freq: " << freq << ", ops.size: " << ops.size() << std::endl;
                    //++it;
                    break;
                }

                default:
                    ++it;
                    break;
            }
    //        std::cout << "Here19: " << ", ops.size: " << ops.size() << std::endl;
            // 如果没有删除操作符，则手动增加迭代器
            ++it;
    //        std::cout << "Here20: " << ", ops.size: " << ops.size() << std::endl;
        } 
    //    std::cout << "Here21: " << ", ops.size: " << ops.size() << std::endl;
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================

        // count the number of times all tensors are used
        std::unordered_map<TensorObj *, size_t> tensorToRefCount;
        // record the memory address offsets of all tensors to be allocated
        std::unordered_map<TensorObj *, size_t> tensorToOffset;

        for (auto &tensor : tensors) {
            tensorToRefCount[tensor.get()] = tensor->getTargets().size();
            // allocate memory for all user-created tensors
            if (tensor.get()->getSource() == nullptr) {
                tensorToOffset[tensor.get()] =
                    allocator.alloc(tensor->getBytes());
            }
        }  

        // traverse in topological order and simulate memory allocation
        for (auto &op : ops) {
            // memory should be allocated for the op's output first
            auto outputs = op->getOutputs();
            for (auto &tensor : outputs) {
                if (tensor) {
                    tensorToOffset[tensor.get()] =
                        allocator.alloc(tensor->getBytes());
                }
            }
            auto inputs = op->getInputs();
            for (auto &tensor : inputs) {
                if (tensor) {
                    auto tensorIter = tensorToRefCount.find(tensor.get());
                    IT_ASSERT(tensorIter != tensorToRefCount.end());
                    IT_ASSERT(tensorToRefCount[tensor.get()] > 0);
                    tensorToRefCount[tensor.get()] -= 1;
                    if (tensorToRefCount[tensor.get()] == 0) {
                        // indicate that this tensor will no longer be used and
                        // perform memory free
                        tensorToRefCount.erase(tensor.get());
                        allocator.free(tensorToOffset[tensor.get()],
                                        tensor->getBytes());
                    }
                }
            }
        }    

        // perform actual memory allocation for non-weight tensors
        for (auto &tensor : tensors) {
            IT_ASSERT(tensorToOffset.find(tensor.get()) !=
                    tensorToOffset.end());
            tensor->setDataBlob(make_ref<BlobObj>(
                tensor->runtime, static_cast<uint8_t *>(allocator.getPtr()) +
                                    tensorToOffset[tensor.get()]));
        }        

        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini