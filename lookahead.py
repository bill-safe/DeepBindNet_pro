import torch
from collections import defaultdict

class Lookahead:
    """
    Lookahead优化器实现 (LookAhead Optimizer: k steps forward, 1 step back)
    
    参考论文: "Lookahead Optimizer: k steps forward, 1 step back"
    https://arxiv.org/abs/1907.08610
    
    参数:
    - optimizer: 内部优化器
    - k: 每k步同步一次慢权重 (默认: 5)
    - alpha: 慢权重更新比例 (默认: 0.5)
    """
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        
        for group in self.param_groups:
            group["counter"] = 0
    
    def update(self, group):
        """
        更新慢权重
        """
        for fast_p in group["params"]:
            if fast_p.grad is None:
                continue
            param_state = self.state[fast_p]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast_p.data)
                param_state["slow_param"].copy_(fast_p.data)
            slow = param_state["slow_param"]
            # 慢权重 = 慢权重 + alpha * (快权重 - 慢权重)
            slow.add_(self.alpha * (fast_p.data - slow))
            # 快权重 = 慢权重
            fast_p.data.copy_(slow)
    
    def update_lookahead(self):
        """
        更新所有组的慢权重
        """
        for group in self.param_groups:
            self.update(group)
    
    def zero_grad(self, set_to_none=False):
        """
        将所有参数的梯度设置为零
        
        参数:
        - set_to_none: 如果为True，则将梯度设置为None而不是0
        """
        return self.optimizer.zero_grad(set_to_none=set_to_none)
    
    def step(self, closure=None):
        """
        执行单步优化
        """
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            group["counter"] += 1
            if group["counter"] % self.k == 0:
                self.update(group)
        return loss
    
    def state_dict(self):
        """
        返回优化器状态字典
        """
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }
    
    def load_state_dict(self, state_dict):
        """
        加载优化器状态字典
        """
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        self.optimizer.load_state_dict(fast_state_dict)
        slow_state_dict = {
            k: v
            for k, v in state_dict["slow_state"].items()
        }
        self.state.update(slow_state_dict)
    
    def add_param_group(self, param_group):
        """
        添加参数组
        """
        param_group["counter"] = 0
        self.optimizer.add_param_group(param_group)
    
    # 转发所有其他属性到内部优化器
    def __getattr__(self, name):
        return getattr(self.optimizer, name)
