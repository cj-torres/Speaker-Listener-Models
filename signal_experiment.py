import torch
import matplotlib.pyplot as plot
import sys
from typing import Sequence
import GaussianLSTM as gl
from rfutils import sliding



class SimpleSpeakerListener(torch.nn.Module):
    """
    This is the simplest verson of the speaker/listener model
    Signals are single draws without a time dimension
    """

    def __init__(self, meanings: int, articulator_dim: int, embedding_sz: int, hidden_sz: int):
        super(SimpleSpeakerListener, self).__init__()
        self.meaning_sz = meanings
        self.articulator_dim = articulator_dim
        #self.embedding_sz = embedding_sz
        self.hidden_sz = hidden_sz
        self.embedding = torch.nn.Embedding(self.meaning_sz, self.hidden_sz)

        # Each distribution is a beta distribution, therefore our output is twice the dimension or number of beta dists
        self.parameter_transform = torch.nn.Parameter(torch.Tensor(hidden_sz, self.articulator_dim * 2))
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                torch.nn.init.xavier_uniform_(p.data)
            else:
                torch.nn.init.zeros_(p.data)

    def forward(self, x):
        embed = self.embedding(x)
        params = torch.nn.functional.softplus(embed @ self.parameter_transform).clamp(min=sys.float_info.min)

        return params

    def dists(self, params):
        dists = []
        for i in range(self.articulator_dim):
            dists.append(torch.distributions.Beta(params[i], params[i + 1]))

        return dists

    def sample(self, dists):
        return [dist.rsample() for dist in dists]

    def sample_calculat_mi(self, x):
        """
        To calculate loss:
        In this model alpha and beta are deterministic, therefore:
        -loop through each possible meaning
        -find alpha and beta
        -calculate likelihood of x
        """
        ms = torch.tensor(list(range(self.meaning_sz))).to(self.device)

        dists = self(ms)
        xs = self.sample(dists)

    def integral_calculate_mi(self):
        dx = .001
        sample = [dx*.1]   # for estimating left-hand side
        sample.extend([i * dx for i in range(1, int(dx ** -1))])
        sample = torch.Tensor(sample)
        ll_m = torch.log(torch.Tensor([self.meaning_sz**-1]))
        x_m = []
        entropy = []
        for m in torch.Tensor(list(range(self.meaning_sz))).int():
            dists = self.dists(self(m))
            # numerical integration without summing up the rectangles; left hand side; estimation for furthest left bar
            x_slices = [beta.log_prob(sample) + torch.log(torch.Tensor([dx])) for beta in dists]
            entropy.append(torch.stack([beta.entropy() for beta in dists],dim=-1).sum(dim=-1))
            # outer "multiply" for log probs
            tensor1 = x_slices[0]
            for x_slice in x_slices[1:]:
                tensor1 = cross_add(tensor1, x_slice)
            x_m.append(tensor1)
        mean_entropy = torch.stack(entropy,dim=-1).mean(dim=-1)
        p_x_m = torch.stack(x_m, dim=-1)                # probability of x given m (before scaled by probability of m)
        #print(p_x_m.size())
        pxm = (p_x_m + ll_m).exp()                        # "multiply" by likelihood of m for p(x,m) (leave log space)
        px = pxm.sum(dim=-1).log()                      # sum along m to yield p(x) (return to log space)
        mi = (pxm*(p_x_m-px.unsqueeze(dim=-1))).sum()   # "divide" p(x|m) by p(x) then multiply by p(x,m) and sum for mi

        return mi, mean_entropy

    def show_tensors(self):
        dx = .001
        sample = [dx*.1]   # for estimating left-hand side
        sample.extend([i * dx for i in range(1, int(dx ** -1))])
        sample = torch.Tensor(sample)
        x_m = []
        for m in torch.Tensor(list(range(self.meaning_sz))).int():
            # numerical integration without summing up the rectangles; left hand side; estimation for furthest left bar
            x_slices = [beta.log_prob(sample) + torch.log(torch.Tensor([dx])) for beta in self.dists(self(m))]

            # outer "multiply" for log probs
            tensor1 = x_slices[0]
            for x_slice in x_slices[1:]:
                tensor1 = cross_add(tensor1, x_slice)
            x_m.append(tensor1.exp()*(10e7))
        for i, signal_space in enumerate(x_m):
            visualize_tensor(signal_space, i)

class SimpleSpeakerListenerLSTM(torch.nn.Module):
    """
    This is the simplest verson of the speaker/listener model
    Signals are single draws without a time dimension
    """

    def __init__(self, meanings: int, articulator_dim: int, embedding_sz: int, hidden_sz: int, num_layers: int,
                 signal_length: int):
        super(SimpleSpeakerListener, self).__init__()
        self.meaning_sz = meanings
        self.articulator_dim = articulator_dim
        self.embedding_sz = embedding_sz
        self.hidden_sz = hidden_sz
        self.num_layers = num_layers
        self.signal_length = signal_length
        self.embedding = torch.nn.Embedding(self.meaning_sz, self.embedding_sz)
        self.lstm = torch.nn.LSTM(
            input_size=self.embedding_sz,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

        self.initial_state = torch.nn.Parameter(torch.stack(
            [torch.randn(self.num_layers, self.hidden_size),
             torch.randn(self.num_layers, self.hidden_size)]
        ))

        # Each distribution is a beta distribution, therefore our output is twice the dimension or number of beta dists
        self.parameter_transform = torch.nn.Parameter(torch.Tensor(hidden_sz, self.articulator_dim * 2))
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                torch.nn.init.xavier_uniform_(p.data)
            else:
                torch.nn.init.zeros_(p.data)

    def get_initial_state(self):
        init_a, init_b = self.initial_state
        return init_a.unsqueeze(dim=1), init_b.unsqueeze(dim=1)

    def forward(self, x):
        embed = self.embedding(x)
        # convert meaning to correct dimension size
        lstm_input = torch.stack([embed]*self.signal_length,dim=0).unsqueeze(dim=0) #1 x S x H
        h0, c0 = self.get_initial_state()
        lstm_output, _ = self.lstm(lstm_input, (h0, c0)).squeeze() #S x H
        params = torch.nn.functional.softplus(lstm_output @ self.parameter_transform) #S x P

        return params

    def dists(self, params):
        """
        given tensor of size S x 2P returns beta parameters S x P
        (Beta takes two parameters)
        """
        dists = []
        for j in range(self.signal_length):
            dist_j = []
            for i in range(self.articulator_dim):
                dist_j.append(torch.distributions.Beta(params[j][i], params[j][i + 1]))
            dists.append(dist_j)

        return dists

    def sample(self, dists):
        return [dist.rsample() for dist in dists]

    def sample_calculat_mi(self, x):
        """
        To calculate loss:
        In this model alpha and beta are deterministic, therefore:
        -loop through each possible meaning
        -find alpha and beta
        -calculate likelihood of x
        """
        ms = torch.tensor(list(range(self.meaning_sz))).to(self.device)

        dists = self(ms)
        xs = self.sample(dists)

    def integral_calculate_mi(self):
        dx = .0001
        sampler = [dx*.1]   # for estimating left-hand side
        sampler.extend([i * dx for i in range(1, int(dx ** -1))])
        sampler = torch.Tensor(sampler)
        ll_m = torch.log(torch.Tensor([self.meaning_sz**-1]))
        x_m = []
        for m in torch.Tensor(list(range(self.meaning_sz))).int():
            # numerical integration without summing up the rectangles; left hand side; estimation for furthest left bar
            dists = self.dists(self(m)) #S x P betas
            signals = [[beta.log_prob(sampler) + torch.log(torch.Tensor([dx])) for beta in signal] for signal in dists]

            # outer "multiply" for log probs
            for signal in signals:
                x_t = []
                tensor1 = signal[0]
                for dim in signal[1:]:
                    tensor1 = cross_add(tensor1, dim)
                x_t.append(tensor1)
            x_m.append(x_t) # check if value is S x (sample_size**dim) tensors

        p_x_m = torch.stack(x_m, dim=-1)                # probability of x given m (before scaled by probability of m)
        #print(p_x_m.size())
        pxm = (p_x_m + ll_m).exp()                        # "multiply" by likelihood of m for p(x,m) (leave log space)
        px = pxm.sum(dim=-1).log()                      # sum along m to yield p(x) (return to log space)
        mi = (pxm*(p_x_m-px.unsqueeze(dim=-1))).sum()   # "divide" p(x|m) by p(x) then multiply by p(x,m) and sum for mi

        return mi

def cross_add(tensor1: torch.Tensor, tensor2: torch.Tensor):
    """
    This simple function does outer addition, outer multiplication in log space
    """
    return tensor1 + torch.transpose(tensor2.repeat(tensor1[...,None].size()), dim0=0, dim1=-1)


def train_simple_model(meanings: int, articulator_dim: int, embedding_sz: int, hidden_sz: int, epochs: int):
    model = SimpleSpeakerListener(meanings, articulator_dim, embedding_sz, hidden_sz)
    optimizer = torch.optim.Adam(params=model.parameters())
    for i in range(epochs):
        optimizer.zero_grad()
        mi, entropy = model.integral_calculate_mi()
        loss = -entropy*.01 - mi
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print("MI loss: " + str(-loss.item()))

    model.show_tensors()

    return model

def visualize_tensor(array: torch.Tensor, i: int):
    numpy_array = array.detach().numpy()
    plot.imshow(numpy_array, cmap='hot', interpolation='nearest')
    plot.savefig("figures\\figure_%s.png" % str(i))
    plot.show(block=False)

