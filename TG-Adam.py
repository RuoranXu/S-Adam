import torch
import torch.nn as nn
import torch.optim as optim
import gudhi as gd
import numpy as np
from sklearn.decomposition import PCA
from collections import deque
import copy

# ----------------------------------------------------------
# 辅助函数 - 拓扑特征计算
# ----------------------------------------------------------

class TopologyFeatureExtractor:
    def __init__(self,
                 n_pca_components=10,       # PCA 降维维度
                 max_history_points=100,    # 缓存多少个模型参数点用于 PH
                 persistence_threshold=0.1, # 过滤掉短生命周期的同调特征
                 r_max_for_rips=None,       # Rips 复形的最大半径，如果 None，则根据数据自动计算
                 use_hessian_for_morse=True # 是否使用海森矩阵计算 Morse 指标
                ):
        self.n_pca_components = n_pca_components
        self.max_history_points = max_history_points
        self.persistence_threshold = persistence_threshold
        self.r_max_for_rips = r_max_for_rips
        self.use_hessian_for_morse = use_hessian_for_morse

        self.history_params = deque(maxlen=max_history_points)
        self.pca = PCA(n_components=n_pca_components)
        self.fitted_pca = False

    def _flatten_params(self, model):
        """展平模型所有参数，并返回一个 numpy 数组"""
        flat_params = []
        for p in model.parameters():
            if p.requires_grad:
                flat_params.append(p.data.cpu().numpy().flatten())
        return np.concatenate(flat_params)

    def _compute_betti_numbers(self, params_points):
        """
        使用持久同调计算 Betti 数。
        params_points: 形状为 (N, D_lowdim) 的 numpy 数组，N 是点数，D_lowdim 是降维后的维度。
        """
        if params_points.shape[0] < 2: # 需要至少两个点来计算持久同调
            return {'b0': 1, 'b1': 0, 'b2': 0} # 默认值

        # 1. 构建 Rips 复形
        #    注意: Rips 复形的半径 r_max_for_rips 需要谨慎选择。
        #    如果 None，gudhi 会根据点集自动选择。
        #    对于参数空间，r_max 可能需要与学习率和参数尺度有关。
        try:
            rips_complex = gd.RipsComplex(points=params_points,
                                          max_edge_length=self.r_max_for_rips)
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=2) # 计算到 dimension 2
        except Exception as e:
            print(f"Error creating Rips complex: {e}. Returning default Betti numbers.")
            return {'b0': 1, 'b1': 0, 'b2': 0}

        # 2. 计算持久同调
        #    这里我们过滤掉生命周期短的同调特征（persistence < threshold）
        #    threshold 0.1 是一个经验值，需要调整。
        try:
            # `persistence` 参数是 gudhi >= 3.x 的用法
            # 如果版本较老，可能需要单独调用 `persistence()` 然后过滤
            # `filtration` 是一个二维列表，每个内层列表代表一个 simplex 的 [顶点索引列表, 过滤值]
            # `filtration` 会按照过滤值排序
            filtered_simplex_tree = simplex_tree.persistence(
                homology_coeff_field=2, # 使用 Z/2Z 系数域
                min_persistence=self.persistence_threshold
            )
            # filtered_simplex_tree = simplex_tree.persistence() # 获取所有
            # persistence_pairs = gd.reduce_persistence_intervals(filtered_simplex_tree,
            #                                                     min_persistence=self.persistence_threshold)

            # 3. 统计 Betti 数
            #    b0: number of connected components (birth=0, death!=inf)
            #    b1: number of cycles/loops (birth > 0, death!=inf)
            #    b2: number of voids/holes (birth > 0, death!=inf)
            b0 = 0
            b1 = 0
            b2 = 0

            for dim, (birth, death) in filtered_simplex_tree:
                if dim == 0 and death - birth >= self.persistence_threshold:
                    b0 += 1
                elif dim == 1 and death - birth >= self.persistence_threshold:
                    b1 += 1
                elif dim == 2 and death - birth >= self.persistence_threshold:
                    b2 += 1

            # 确保至少有一个连通组件 (b0 >= 1)
            if b0 == 0: b0 = 1

            return {'b0': b0, 'b1': b1, 'b2': b2}

        except Exception as e:
            print(f"Error computing persistence or filtering: {e}. Returning default Betti numbers.")
            return {'b0': 1, 'b1': 0, 'b2': 0}


    def _compute_morse_indicator(self, model, loss_fn, optimizer_state, data_loader=None, sample_data=None):
        """
        尝试使用海森矩阵计算 Morse 指标。
        这是一个计算密集型操作，通常只在关键时刻（如 epoch 结束，梯度接近零）进行。
        """
        if not self.use_hessian_for_morse:
            # 简化版：用梯度范数或直接硬编码
            # 这里的逻辑依然非常简化，仅为演示
            current_grad_norm = torch.norm(torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])).item()
            # 梯度很小，可能在临界点附近
            if current_grad_norm < 1e-5:
                return 0 # 假设是局部最小值或平坦区域
            else:
                return 1 # 假设还在下降路径上

        # ----------------------------------------------------------
        # 复杂的海森矩阵计算
        # ----------------------------------------------------------
        # 需要一个小的 batch 的数据来计算海森矩阵。
        # 理想情况下，应该在训练的某个检查点进行。
        # 这里我们假设 data_loader 和 sample_data 已经准备好。

        if sample_data is None:
            print("Warning: Morse indicator computation requires sample_data.")
            return 1 # Fallback value

        # 确保计算图可以跟踪梯度
        for p in model.parameters():
            p.requires_grad_(True)

        try:
            # 1. 获取一个 batch 的数据
            #    如果 data_loader 提供，则取第一个 batch
            #    否则，如果 sample_data 是一个 tensor，则直接使用
            if data_loader:
                try:
                    data_batch, target_batch = next(iter(data_loader))
                    data_batch = data_batch.to(next(model.parameters()).device)
                    target_batch = target_batch.to(next(model.parameters()).device)
                except StopIteration:
                    print("Warning: DataLoader is empty. Cannot compute Hessian.")
                    return 1
            elif isinstance(sample_data, torch.Tensor):
                # 假设 sample_data 已经是准备好的 batch
                data_batch = sample_data.to(next(model.parameters()).device)
                # 假设 loss_fn 可以处理这个 tensor，或者 target 是固定的
                # 这是一个非常粗略的假设，实际需要根据 loss_fn 和 model 调整
                target_batch = torch.randint(0, 10, (data_batch.shape[0],)).to(next(model.parameters()).device) # 示例
            else:
                 print("Warning: Invalid sample_data format. Cannot compute Hessian.")
                 return 1

            # 2. 计算损失
            outputs = model(data_batch)
            loss = loss_fn(outputs, target_batch)

            # 3. 计算梯度 (第一阶导数)
            model.zero_grad()
            loss.backward(create_graph=True) # create_graph=True to allow second-order differentiation

            # 4. 计算海森矩阵 (第二阶导数)
            #     Hesse = d^2 L / d(params)^2
            #    这是一个高度计算密集的操作，特别是对于大型模型。
            #    通常只对模型参数的一个小子集进行估计，或者使用近似方法。
            #    为了简化，我们假设所有参数都需要计算。
            hessian_params = []
            for p in model.parameters():
                if p.grad is not None:
                    # d(dL/dpi) / dpj
                    # 我们需要对每个参数 p_j, 计算 L 关于 p_i 的导数 (p.grad) 对 p_j 的导数。
                    # 这通常需要 iterate over all parameters.
                    # 一个更高效的实现是利用 autograd.functional.hessian，但它对模型参数结构要求很高。
                    # 这里是一个概念性的表示，实际实现会非常复杂。
                    # 假设我们有一个函数 `compute_full_hessian(loss, model.parameters())`
                    # `hessian_params.append(torch.autograd.grad(p.grad, p, retain_graph=True))` # THIS IS WRONG
                    pass # Placeholder for actual Hessian computation

            # 假设我们已经计算出了完整的海森矩阵 `full_hessian` (N x N, N=total_params)
            # full_hessian = compute_full_hessian(loss, model.parameters()) # Placeholder

            # -------------------------------------------------------------------
            #  注意: 计算完整的全局海森矩阵通常是不可行的。
            #  实际方法可能包括:
            #  a) 随机部分海森矩阵 (Randomized Hessian Approximation)
            #  b) K-FAC (Kronecker-factored Approximate Curvature)
            #  c) 只考虑对角线海森矩阵 (Diagonal Hessian Approximation)
            #  d) 估算局部曲率 (e.g., using finite differences on gradients)
            # -------------------------------------------------------------------

            # for simplicity, let's assume we *can* get the eigenvalues
            # For demonstration, let's assume `hessian_values` is a list/tensor of eigenvalues
            # `morse_indicator` = number of negative eigenvalues
            # `morse_indicator` = torch.sum(torch.tensor(hessian_values) < 0).item() if hessian_values else 0

            # Fallback for demonstration if Hessian computation is too complex
            if 'full_hessian' not in locals():
                print("Hessian computation not implemented in detail. Using fallback for Morse indicator.")
                return 1 # Placeholder value

            # Example: If we somehow got eigenvalues from full_hessian
            # morse_indicator = torch.sum(torch.linalg.eigvalsh(full_hessian) < 0).item()

            # For now, let's return a placeholder
            print("Hessian computation placeholder.")
            return 1 # Placeholder value

        except Exception as e:
            print(f"Error computing Morse indicator with Hessian: {e}. Returning fallback.")
            # Fallback value if Hessian computation fails
            return 1 # Placeholder

        finally:
            # 确保 grad 标志被正确设置
            for p in model.parameters():
                p.requires_grad_(p.requires_grad) # Restore original requires_grad

    def update_history(self, model):
        """将当前模型参数添加到历史记录中"""
        param_state = self._flatten_params(model)
        self.history_params.append(param_state)

        # 拟合 PCA 模型，如果历史点足够多且 PCA 未拟合
        if len(self.history_params) >= self.max_history_points and not self.fitted_pca:
            history_array = np.array(list(self.history_params))
            try:
                self.pca.fit(history_array)
                self.fitted_pca = True
                print("PCA fitted with historical parameters.")
            except Exception as e:
                print(f"Error fitting PCA: {e}")
                self.fitted_pca = False # 允许重试

    def get_topology_features(self, model, loss_fn, optimizer_state, data_loader=None, sample_data=None):
        """计算并返回 Betti 数和 Morse 指标"""

        # 1. 更新历史参数和 PCA 模型
        self.update_history(model)

        # 2. 计算 Betti 数
        betti_numbers = {'b0': 1, 'b1': 0, 'b2': 0}
        if self.fitted_pca and len(self.history_params) >= 2: # PCA 必须已拟合，且至少有两个点
            try:
                history_array = np.array(list(self.history_params))
                # 使用已拟合的 PCA 进行降维
                low_dim_params = self.pca.transform(history_array)
                betti_numbers = self._compute_betti_numbers(low_dim_params)
            except Exception as e:
                print(f"Error in Betti number calculation after PCA transform: {e}")
                # Fallback to default

        # 3. 计算 Morse 指标
        #    Morse 指标计算可能很昂贵，仅在需要时或在特定触发器上计算
        #    这里我们将其设置为可选
        morse_indicator = 1 # Default value
        if self.use_hessian_for_morse:
            # 实际应用中，Morse 指标的计算可能只在 epoch 结束时，或者梯度范数低于某个阈值时进行
            # 并且需要传入 model, loss_fn, optimizer_state, data_loader, sample_data
            morse_indicator = self._compute_morse_indicator(model, loss_fn, optimizer_state, data_loader, sample_data)
        else:
            # 简化版 Morse 指标 (仅使用梯度范数，如原版)
            grad_norm = torch.norm(torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])).item()
            morse_indicator = 0 if grad_norm < 1e-5 else 1


        print(f"Topology Features: Betti = {betti_numbers}, Morse = {morse_indicator}")
        return betti_numbers, morse_indicator

# ----------------------------------------------------------
# 拓扑引导的 Adam 优化器
# ----------------------------------------------------------

class TopologyGuidedAdam(optim.Adam):
    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0,
                 amsgrad=False,
                 # 拓扑引导的超参数
                 betti_thresholds={'b0': 5, 'b1': 3, 'b2': 1}, # 阈值，根据 Betti 数不同维度设定
                 morse_threshold=1,       # Morse 指标的阈值
                 # TopologyFeatureExtractor 的参数
                 n_pca_components=10,
                 max_history_points=100,
                 persistence_threshold=0.1,
                 r_max_for_rips=None,
                 use_hessian_for_morse=True, # 控制是否使用昂贵的海森计算
                 # 触发拓扑特征计算的策略
                 topology_update_freq_iter=100, # 每多少次迭代计算一次拓扑特征
                 topology_update_freq_epoch=1 # 每多少个 epoch 计算一次拓扑特征
                ):
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad)

        self.betti_thresholds = betti_thresholds
        self.morse_threshold = morse_threshold

        self.topology_extractor = TopologyFeatureExtractor(
            n_pca_components=n_pca_components,
            max_history_points=max_history_points,
            persistence_threshold=persistence_threshold,
            r_max_for_rips=r_max_for_rips,
            use_hessian_for_morse=use_hessian_for_morse
        )

        # 状态追踪
        self.current_iter = 0
        self.current_epoch = 0 # 需要在外部传递
        self.topology_update_freq_iter = topology_update_freq_iter
        self.topology_update_freq_epoch = topology_update_freq_epoch
        self.last_topology_update_iter = -float('inf')
        self.last_topology_update_epoch = -float('inf')

        # 缓存当前的拓扑特征，避免重复计算
        self.cached_betti_numbers = {'b0': 1, 'b1': 0, 'b2': 0}
        self.cached_morse_indicator = 1
        self.last_calculated_iter = -1

        # 记录原始 beta1 和 eps，用于恢复
        self.original_betas = list(betas)
        self.original_eps = eps

        # 存储用于 Morse 指标计算的上下文（模型，损失函数，数据）
        # 这些需要在 step 函数中被动态传入，或者作为参数传给优化器
        self._context = {
            'model': None,
            'loss_fn': None,
            'optimizer_state': None,
            'data_loader': None,
            'sample_data': None
        }

    def set_context(self, model, loss_fn, optimizer_state, data_loader=None, sample_data=None):
        """设置计算拓扑特征所需的上下文信息"""
        self._context['model'] = model
        self._context['loss_fn'] = loss_fn
        self._context['optimizer_state'] = optimizer_state
        self._context['data_loader'] = data_loader
        self._context['sample_data'] = sample_data

    def _check_topology_update(self):
        """判断是否需要更新拓扑特征"""
        if self.current_epoch > self.last_topology_update_epoch + self.topology_update_freq_epoch - 1:
             # 检查是否是 epoch 更新周期
             return True
        if self.current_iter > self.last_topology_update_iter + self.topology_update_freq_iter - 1:
             # 检查是否是迭代更新周期
             return True
        return False

    def _update_topology_features(self):
        """实际计算并缓存拓扑特征"""
        if self._context['model'] is None:
            print("Warning: Topology context (model, loss_fn, etc.) not set. Skipping topology feature update.")
            return False

        model = self._context['model']
        loss_fn = self._context['loss_fn']
        optimizer_state = self._context['optimizer_state'] # Not used by extractor, but passed for completeness
        data_loader = self._context['data_loader']
        sample_data = self._context['sample_data']

        # 检查模型参数是否有梯度，没有梯度可能无法计算很多东西
        if not any(p.grad is not None for p in model.parameters()):
            print("Warning: No gradients found for model parameters. Cannot compute topology features reliably.")
            # In this case, we might want to skip updating topology features, or use default values.
            # For now, let's just return False indicating no update happened.
            return False

        try:
            betti_numbers, morse_indicator = self.topology_extractor.get_topology_features(
                model, loss_fn, optimizer_state, data_loader, sample_data
            )
            self.cached_betti_numbers = betti_numbers
            self.cached_morse_indicator = morse_indicator
            self.last_topology_update_iter = self.current_iter
            self.last_topology_update_epoch = self.current_epoch
            self.last_calculated_iter = self.current_iter # Mark that calculation happened at this iter
            return True
        except Exception as e:
            print(f"Error during topology feature calculation: {e}")
            return False

    def _apply_topology_guidance(self):
        """根据缓存的拓扑特征动态调整 Adam 的超参数"""
        betti = self.cached_betti_numbers
        morse = self.cached_morse_indicator

        # ----------------------------------------------------------
        # 调整 Betas (动量)
        # ----------------------------------------------------------
        # 逻辑：
        # - 如果 Betti 数高，意味着复杂地形 -> 减小 beta1 (更关注当前梯度)
        # - 如果 Betti 数低，意味着平坦地形 -> 增大 beta1 (利用历史信息加速)
        # - 阈值需要根据 Betti 数的维度来考虑

        beta1 = self.original_betas[0] # Start with original beta1

        # Example logic: more complex landscape -> lower momentum
        # This is a heuristic. The exact mapping needs experimentation.

        # Higher Betti numbers generally indicate more complex structure.
        # We can combine the Betti numbers or use them individually.

        # Simple combined heuristic:
        complexity_score = (betti.get('b0', 1) * 2 + # b0 higher means more components, potentially complex
                            betti.get('b1', 0) * 5 + # b1 higher means more loops, very complex
                            betti.get('b2', 0) * 10) # b2 higher means more voids, also very complex

        # Adjusting beta1 based on a combined score or individual thresholds
        # This mapping is very experimental:
        if complexity_score > sum(self.betti_thresholds.values()): # If complexity is generally high
            beta1 = max(0.5, self.original_betas[0] * 0.8) # Reduce momentum significantly
        elif complexity_score < 3: # If complexity is very low
            beta1 = min(0.99, self.original_betas[0] * 1.05) # Increase momentum slightly
        else:
            # Intermediate complexity, try to use original or slightly adjusted
            # We can also use the individual thresholds to refine
            if betti.get('b1', 0) > self.betti_thresholds['b1']: # High number of loops
                beta1 = max(0.5, self.original_betas[0] * 0.9)
            if betti.get('b2', 0) > self.betti_thresholds['b2']: # High number of voids
                beta1 = max(0.5, self.original_betas[0] * 0.85)


        # Ensure beta1 is within reasonable bounds
        beta1 = max(0.1, min(0.999, beta1))

        # ----------------------------------------------------------
        # 调整 Epsilon (稳定性参数)
        # ----------------------------------------------------------
        # 逻辑：
        # - 如果 Morse 指标高（或海森负特征值多），意味着鞍点多 -> 减小 eps (增加稳定性)
        # - 如果 Morse 指标低（局部最小值/平坦），意味着稳定 -> 增大 eps (更新更直接)

        eps = self.original_eps
        if morse > self.morse_threshold:
            eps = max(1e-10, self.original_eps / 10.0) # Reduce eps
        else:
            eps = min(1e-5, self.original_eps * 10.0) # Increase eps

        # Ensure eps is within reasonable bounds
        eps = max(1e-10, min(1e-5, eps))


        # ----------------------------------------------------------
        # 应用调整后的超参数
        # ----------------------------------------------------------
        for group in self.param_groups:
            group['betas'] = (beta1, group['betas'][1]) # Only adjust beta1
            group['eps'] = eps

        print(f"Applied Topology Guidance: New betas={self.param_groups[0]['betas']}, New eps={self.param_groups[0]['eps']}")


    def step(self, closure=None):
        """
        执行一步优化。
        在此方法中，我们将计算拓扑特征（如果需要），然后应用调整后的超参数。
        """
        if closure is not None:
            # If closure is provided, we need to call it first to get gradients
            loss = closure()
            # After closure, gradients are available.
            # We still need model context if closure doesn't provide it directly.
            # For simplicity, let's assume context is already set via set_context.
            # But if closure itself computes loss and backward, it might manage its own context.
            # We proceed assuming gradients are populated and context is set.
        else:
            # If no closure, assume gradients are already computed before calling step.
            # We still need context.
            pass # Gradients are assumed to be available

        # 1. 检查是否需要更新拓扑特征
        #    This check is based on iter and epoch counts managed by the user.
        #    The user is responsible for incrementing current_iter and current_epoch.
        if self._check_topology_update():
            if self._update_topology_features(): # Attempt to update and cache
                # 2. 应用拓扑引导
                self._apply_topology_guidance()
            else:
                # If topology update failed, we might want to reset to default or keep old values.
                # For safety, let's reset to original values if calculation failed.
                print("Topology feature calculation failed. Reverting to original Adam parameters.")
                self.reset_to_original_params()
        else:
            # If not due for update, but we have cached features, apply them again.
            # This ensures guidance is applied at every step *after* the first calculation.
            if self.last_calculated_iter != -1: # If features have been calculated at least once
                self._apply_topology_guidance()
            else:
                # If features haven't been calculated yet, just do a standard Adam step.
                # This might happen on the very first few steps before the update frequency is met.
                pass


        # 3. 执行标准的 Adam 更新
        #    super().step() uses the potentially modified param_groups
        return super().step(closure)

    def reset_to_original_params(self):
        """将优化器参数重置回初始值"""
        for group in self.param_groups:
            group['betas'] = (self.original_betas[0], self.original_betas[1])
            group['eps'] = self.original_eps

    def zero_grad(self):
        """清零所有模型参数的梯度"""
        super().zero_grad()

    # Add methods to manage epoch and iteration counts for frequency control
    def increment_iter(self):
        self.current_iter += 1

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        self.last_topology_update_epoch = epoch # Reset epoch tracker when a new epoch starts

# ----------------------------------------------------------
# 使用示例 (示意)
# ----------------------------------------------------------
if __name__ == "__main__":
    # 1. 定义一个简单的模型
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(10, 20)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(20, 2)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    model = SimpleNet()
    loss_fn = nn.CrossEntropyLoss()
    data_loader = None # Placeholder for real data loader
    sample_data = torch.randn(32, 10) # A sample batch of data

    # 2. 初始化拓扑引导的 Adam 优化器
    #    可以根据需要调整 TopologyFeatureExtractor 的参数
    optimizer = TopologyGuidedAdam(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
        # 拓扑引导超参数
        betti_thresholds={'b0': 5, 'b1': 3, 'b2': 1},
        morse_threshold=1,
        # Extractor 参数
        n_pca_components=10,      # 降维到 10 维
        max_history_points=100,   # 缓存 100 个参数点
        persistence_threshold=0.1, # PH 过滤阈值
        use_hessian_for_morse=False, # 初始阶段使用简化版 Morse 计算 (避免复杂实现)
        topology_update_freq_iter=50, # 每 50 迭代计算一次拓扑特征
        topology_update_freq_epoch=2 # 每 2 个 epoch 计算一次拓扑特征
    )

    # 3. 设置优化器上下文 (非常重要!)
    #    这使得优化器能够访问模型、损失函数等必要信息
    optimizer.set_context(model, loss_fn, optimizer.state, data_loader=None, sample_data=sample_data)


    # 4. 模拟训练循环
    num_epochs = 5
    steps_per_epoch = 100
    total_iters = num_epochs * steps_per_epoch

    print("Starting simulated training...")

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        optimizer.set_epoch(epoch) # Set current epoch for frequency control

        for step in range(steps_per_epoch):
            optimizer.increment_iter() # Increment iteration count

            # 模拟数据加载和计算损失
            # 假设在调用 step 之前，模型已经进行了 forward pass 并且计算了 loss
            # 并且 loss.backward() 已经调用。
            # ----------------------------------------------------------
            # 模拟 Forward Pass, Loss Calculation, Backward Pass
            # ----------------------------------------------------------
            optimizer.zero_grad() # Clear previous gradients
            inputs = torch.randn(32, 10)
            labels = torch.randint(0, 2, (32,))
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward() # Compute gradients

            # ----------------------------------------------------------
            # 调用 optimizer.step()
            # ----------------------------------------------------------
            # 优化器内部会根据拓扑特征调整参数，然后执行更新
            optimizer.step()

            if (optimizer.current_iter) % 20 == 0:
                print(f"Iter [{optimizer.current_iter}/{total_iters}] Loss: {loss.item():.4f}")

        # ----------------------------------------------------------
        # Epoch 结束时的额外处理（可选）
        # ----------------------------------------------------------
        # 例如，可以在这里强制计算一次拓扑特征，或者进行模型评估
        # optimizer.set_epoch(epoch + 1) # Ensure epoch count is correctly updated before next epoch starts
        # If you want epoch-based updates to trigger, ensure epoch count is correct.

    print("\nSimulated training finished.")

    # ----------------------------------------------------------
    # 注意:
    # 1.  `_compute_betti_numbers` 中的 `r_max_for_rips` 和 `persistence_threshold`
    #     需要仔细调整，它们对结果影响很大。
    # 2.  `_compute_morse_indicator` 中的海森矩阵计算非常复杂，
    #     在实际应用中可能需要采用近似方法 (如 K-FAC, Random Hessian) 或
    #     仅在特定条件下（如梯度很小）触发。
    # 3.  `use_hessian_for_morse=False` 是为了让代码能跑起来，
    #     但完整的拓扑引导需要准确的 Morse 指标。
    # 4.  `topology_update_freq_iter` 和 `topology_update_freq_epoch`
    #     需要根据计算开销和收敛需求来平衡。
    # 5.  "终极版" 意味着我们设计了一个更完整、更具潜力的框架。
    #     真正的 "终极" 需要通过大量实验来验证和细化。
    # ----------------------------------------------------------
