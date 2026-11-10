"""Time-major GPU ring, samples never cross reset or fallen transitions."""

import torch


class Replay:
    def __init__(self, n, capacity, device, sender_memory_dim=0):
        self.capacity = capacity
        self.n = n
        self.ptr = 0
        self.count = 0
        self.device = device
        self.s = torch.empty(capacity, n, 160, device=device)
        self.a = torch.empty(capacity, n, 62, device=device)
        self.h = torch.empty(capacity, n, 324, device=device)
        self.done = torch.ones(capacity, n, device=device, dtype=torch.bool)
        self.sender = (torch.empty(capacity, n, sender_memory_dim, device=device)
                       if sender_memory_dim else None)

    def add(self, s, a, h, done, sender=None):
        i = self.ptr
        self.s[i].copy_(s)
        self.a[i].copy_(a)
        self.h[i].copy_(h)
        self.done[i].copy_(done)
        if self.sender is not None:
            if sender is None:
                raise ValueError("Explicit sender memory required for this replay")
            self.sender[i].copy_(sender)
        self.ptr = (i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample(self, batch, length, with_sender=False):
        # length+1 states, length actions; valid windows must be consecutive within episode.
        if self.count <= length:
            raise ValueError("Replay warmup too short")
        ids = torch.randint(self.n, (batch * 3,), device=self.device)
        start = torch.randint(self.count - length, (batch * 3,), device=self.device)
        t = (
            self.ptr
            - self.count
            + start[:, None]
            + torch.arange(length + 1, device=self.device)
        ) % self.capacity
        ok = ~self.done[t[:, :-1], ids[:, None]].any(-1)
        pick = ok.nonzero().flatten()[:batch]
        if len(pick) < batch:
            raise RuntimeError("Too few valid replay windows")
        t = t[pick]
        ids = ids[pick]
        result = (
            self.s[t, ids[:, None]],
            self.a[t[:, :-1], ids[:, None]],
            self.h[t[:, 0], ids],
        )
        if with_sender:
            if self.sender is None:
                raise ValueError("Replay has no sender memory")
            return (*result, self.sender[t[:, 0], ids])
        return result
