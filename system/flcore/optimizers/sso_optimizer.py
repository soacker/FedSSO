from functools import reduce

import torch
from torch.optim import Optimizer

class sso_optimizer(Optimizer):
    '''
    arg:
        lr: 全局学习率
        history_size: 曲率信息长度，当使用全部信息时，则将其设定为 global_rounds 值，此时对应 BFGS 构造的逆形式。
        enforce: 强制正定性策略：目前提供2种方案，=0， 为第一种，是只添加满足曲率条件的曲率。=1，为第二种，通过强制修正曲率。
        sk_sigma: 用于修正 s = x_k_1 - x_k 时，存在的误差，建议值 0.7，或1
        lamb_,Lamb_: 当 enforce=1，曲率条件的上界和下界。
        all: 使用全部梯度时，all=0; 当使用平均梯度时，all=1。理论上，all=1时，更稳定一些，当面对较复杂的非凸模型时。
        local_steps: client侧的局部更新步数，配合平均梯度的使用。
    '''
    def __init__(self,
                 params,
                 lr=1,
                 history_size=100,
                 enforce=0,
                 sk_sigma=0.7,
                 lamb_=0.0001,
                 Lamb_=10000,
                 all=0,
                 local_steps=10):
        defaults = dict(
            lr=lr,
            history_size=history_size)
        super(sso_optimizer, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("param_groups error")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None
        self.enforce = enforce
        self.sk_sigma = sk_sigma
        self.lamb_ = lamb_
        self.Lamb_ = Lamb_
        self.all = all
        self.local_steps = local_steps

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        if self.all == 0:
            return torch.cat(views, 0)
        elif self.all == 1:
            return torch.cat(views, 0) / self.local_steps
        else:
            raise ValueError("no implement global gradient")

    # 进行梯度下降
    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            p.add_(update[offset:offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata)

    def _directional_evaluate(self, closure, x, t, d):
        self._add_grad(t, d)
        loss = float(closure())
        flat_grad = self._gather_flat_grad()
        self._set_param(x)
        return loss, flat_grad

    @torch.no_grad()
    def step(self):
        assert len(self.param_groups) == 1

        group = self.param_groups[0]
        lr = group['lr']
        history_size = group['history_size']

        state = self.state[self._params[0]]
        state.setdefault('n_iter', 0)

        flat_grad = self._gather_flat_grad()

        # tensors cached in state (for tracing)
        d = state.get('d')
        t = state.get('t')
        old_dirs = state.get('old_dirs')
        old_stps = state.get('old_stps')
        ro = state.get('ro')
        H_diag = state.get('H_diag')
        prev_flat_grad = state.get('prev_flat_grad')


        state['n_iter'] += 1

        ############################################################
        # compute gradient descent direction
        ############################################################
        if state['n_iter'] == 1:
            d = flat_grad.neg()
            old_dirs = []
            old_stps = []
            ro = []
            H_diag = 1
        else:
            # do lbfgs update (update memory)
            y = flat_grad.sub(prev_flat_grad)


            s = d.mul(t)
            s = s * self.sk_sigma

            ys = y.dot(s)  # y*s
            cur = s.dot(s) / ys

            if self.enforce == 2:
                lamb_ = self.lamb_
                Lamb_ = self.Lamb_
                if cur > lamb_ and cur < Lamb_:
                    # updating memory
                    if len(old_dirs) == history_size:
                        # shift history by one (limited-memory)
                        old_dirs.pop(0)
                        old_stps.pop(0)
                        ro.pop(0)

                    # store new direction/step
                    old_dirs.append(y)
                    old_stps.append(s)
                    ro.append(1. / ys)

                    # update scale of initial Hessian approximation
                    H_diag = ys / y.dot(y)  # (y*y)
                else:
                    print("cur condition not satisfy, fixed y")

                    if len(old_dirs) == history_size:
                        # shift history by one (limited-memory)
                        old_dirs.pop(0)
                        old_stps.pop(0)
                        ro.pop(0)

                    # y = s * 2 / (lamb_ + Lamb_)
                    s = y * 2 / (lamb_ + Lamb_)

                    ys = y.dot(s)  # y*s

                    # store new direction/step
                    old_dirs.append(y)
                    old_stps.append(s)
                    ro.append(1. / ys)

                    # update scale of initial Hessian approximation
                    H_diag = ys / y.dot(y)  # (y*y)
            elif self.enforce == 1:
                if ys > 1e-10:
                    # updating memory
                    if len(old_dirs) == history_size:
                        # shift history by one (limited-memory)
                        old_dirs.pop(0)
                        old_stps.pop(0)
                        ro.pop(0)

                    # store new direction/step
                    old_dirs.append(y)
                    old_stps.append(s)
                    ro.append(1. / ys)

                    # update scale of initial Hessian approximation
                    H_diag = ys / y.dot(y)  # (y*y)
                else:
                    print("cur condition not satisfy")
                    if len(old_dirs) == history_size:
                        # shift history by one (limited-memory)
                        old_dirs.pop(0)
                        old_stps.pop(0)
                        ro.pop(0)
                    y = old_dirs[-1]
                    s = old_stps[-1]

                    old_dirs.append(y)
                    old_stps.append(s)
                    ro.append(1. / ys)
                    H_diag = ys / y.dot(y)  # (y*y)

            elif self.enforce == 0:
                if ys > 1e-10:
                    # updating memory
                    if len(old_dirs) == history_size:
                        # shift history by one (limited-memory)
                        old_dirs.pop(0)
                        old_stps.pop(0)
                        ro.pop(0)

                    # store new direction/step
                    old_dirs.append(y)
                    old_stps.append(s)
                    ro.append(1. / ys)

                    # update scale of initial Hessian approximation
                    H_diag = ys / y.dot(y)  # (y*y)
                else:
                    print("cur condition not satisfy")

            else:
                raise ValueError("no implement enforce method")

            # compute the approximate (L-BFGS) inverse Hessian
            # multiplied by the gradient
            num_old = len(old_dirs)

            if 'al' not in state:
                state['al'] = [None] * history_size
            al = state['al']

            # iteration in L-BFGS loop collapsed to use just one buffer
            q = flat_grad.neg()
            for i in range(num_old - 1, -1, -1):
                al[i] = old_stps[i].dot(q) * ro[i]
                q.add_(old_dirs[i], alpha=-al[i])

            # multiply by initial Hessian
            # r/d is the final direction
            d = r = torch.mul(q, H_diag)
            for i in range(num_old):
                be_i = old_dirs[i].dot(r) * ro[i]
                r.add_(old_stps[i], alpha=al[i] - be_i)

        if prev_flat_grad is None:
            prev_flat_grad = flat_grad.clone(memory_format=torch.contiguous_format)
        else:
            prev_flat_grad.copy_(flat_grad)

        ############################################################
        # compute step length
        ############################################################
        # reset initial guess for step size
        if state['n_iter'] == 1:
            t = min(1., 1. / flat_grad.abs().sum()) * lr
        else:
            t = lr

        # 进行梯度下降
        self._add_grad(t, d)


        state['d'] = d
        state['t'] = t
        state['old_dirs'] = old_dirs
        state['old_stps'] = old_stps
        state['ro'] = ro
        state['H_diag'] = H_diag
        state['prev_flat_grad'] = prev_flat_grad

        return