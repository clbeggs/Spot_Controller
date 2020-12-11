import torch

mu1 = torch.rand((4))
cov1 = torch.ones((4))

a = torch.distributions.Normal(mu1, cov1)

mu2 = torch.rand((4))
cov2 = torch.ones((4))
b = torch.distributions.Normal(mu2, cov2)
print(a)
print(b)

bb = b.sample()
aa = a.sample()

kl = torch.distributions.kl.kl_divergence(aa, bb)
print(kl)

#print(kl_divergence(a, b))
