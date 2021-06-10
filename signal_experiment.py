import torch
import matplotlib.pyplot as plot
import sys
import pyro
from typing import Sequence
import GaussianLSTM as gl
from rfutils import sliding


class PyroSimpleSpeakerListener(pyro.nn.PyroModule):
    """
    This is the simplest verson of the speaker/listener model
    Signals are single draws without a time dimension
    """

    def __init__(self, meanings: int, articulator_dim: int, embedding_sz: int, hidden_sz: int):
        super(PyroSimpleSpeakerListener, self).__init__()
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
        for j in range(self.meaning_sz):
            m_dist = []
            for i in range(self.articulator_dim):
                mx_beta = pyro.distributions.Beta(params[j][2*i], params[j][2*i + 1])
                m_dist.append(mx_beta)
            dists.append(m_dist)
        return dists

    def sample(self, dists):
        return [[beta.rsample() for beta in dist] for dist in dists]

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


def integral_calculate_mi(speaker_listener_model):
    dx = .001
    sample = [dx*.1]   # for estimating left-hand side
    sample.extend([i * dx for i in range(1, int(dx ** -1))])
    sample = torch.Tensor(sample)
    ll_m = torch.log(torch.Tensor([speaker_listener_model.meaning_sz**-1]))
    entropy = []
    meanings = torch.Tensor(range(speaker_listener_model.meaning_sz)).int()
    dists = speaker_listener_model.dists(speaker_listener_model(meanings))
    # numerical integration without summing up the rectangles; left hand side; estimation for furthest left bar
    x_slices = []

    for dist in dists:
        samples = [beta.log_prob(sample) + torch.log(torch.Tensor([dx])) for beta in dist]
        entropy.append(torch.stack([beta.entropy() for beta in dist], dim=-1).sum(dim=-1))
        # outer "multiply" for log probs
        full_tensor = samples[0]
        for x_dim in samples[1:]:
            full_tensor = cross_add(full_tensor, x_dim)
        x_slices.append(full_tensor)
    p_x_m = torch.stack(x_slices, dim=-1)           # probability of x given m (before scaled by probability of m)

    mean_entropy = torch.stack(entropy,dim=-1).mean(dim=-1)
    pxm = (p_x_m + ll_m).exp()                        # "multiply" by likelihood of m for p(x,m) (leave log space)
    px = pxm.sum(dim=-1).log()                      # sum along m to yield p(x) (return to log space)
    mi = (pxm*(p_x_m-px.unsqueeze(dim=-1))).sum()   # "divide" p(x|m) by p(x) then multiply by p(x,m) and sum for mi

    return mi, mean_entropy



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
            x_m.append(tensor1.exp()*(10e9))
        for i, signal_space in enumerate(x_m):
            visualize_tensor(signal_space, i)

class PyroSimpleSpeakerListenerLSTM(torch.nn.Module):
    """
    This is the simplest verson of the speaker/listener model
    Signals are single draws without a time dimension
    """

    def __init__(self, meaning_sz: int, articulator_dim: int, embedding_sz: int, hidden_sz_1: int, hidden_sz_2: int,
                 num_layers: int, signal_length: int):
        super(PyroSimpleSpeakerListenerLSTM, self).__init__()
        self.meaning_sz = meaning_sz
        self.articulator_dim = articulator_dim
        self.embedding_sz = embedding_sz
        self.hidden_sz_1 = hidden_sz_1
        self.hidden_sz_2 = hidden_sz_2
        self.num_layers = num_layers
        self.signal_length = signal_length
        self.embedding = torch.nn.Embedding(self.meaning_sz, self.embedding_sz)
        self.encoder_lstm = torch.nn.LSTM(
            input_size=self.embedding_sz,
            hidden_size=self.hidden_sz_1,
            num_layers=self.num_layers,
            batch_first=False
        )

        self.encoder_initial_state = torch.nn.Parameter(torch.stack(
            [torch.randn(self.num_layers, self.hidden_sz_1),
             torch.randn(self.num_layers, self.hidden_sz_1)]
        ))

        self.decoder_lstm = torch.nn.LSTM(
            input_size=self.articulator_dim,
            hidden_size=self.hidden_sz_2,
            num_layers=self.num_layers,
            batch_first=False
        )

        self.decoder_initial_state = torch.nn.Parameter(torch.stack(
            [torch.randn(self.num_layers, self.hidden_sz_2),
             torch.randn(self.num_layers, self.hidden_sz_2)]
        ))

        # Each distribution is a beta distribution, therefore our output is twice the dimension or number of beta dists
        self.parameter_transform = torch.nn.Parameter(torch.Tensor(hidden_sz_1, self.articulator_dim * 2))
        # Returns matrix of correct size for softmax
        self.to_softmax = torch.nn.Parameter(torch.Tensor(self.hidden_sz_2, self.meaning_sz))
        self.predictor = torch.nn.Softmax(dim=-1)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                torch.nn.init.xavier_uniform_(p.data)
            else:
                torch.nn.init.zeros_(p.data)

    def get_encoder_initial_state(self, batch_size):
        init_a, init_b = self.encoder_initial_state
        return torch.stack([init_a]*batch_size,dim=1), torch.stack([init_b]*batch_size,dim=1)

    def get_decoder_initial_state(self, batch_size):
        init_a, init_b = self.decoder_initial_state
        return torch.stack([init_a]*batch_size,dim=1), torch.stack([init_b]*batch_size,dim=1)

    def encoder(self, x):
        embed = self.embedding(x)
        # convert meaning to correct dimension size
        lstm_input = torch.stack([embed]*self.signal_length,dim=0)                      # S x B x H
        h0, c0 = self.get_encoder_initial_state(len(x))
        lstm_output, _ = self.encoder_lstm(lstm_input, (h0, c0))
        params = torch.nn.functional.softplus(lstm_output @ self.parameter_transform)   # S x B x P
        #print(params)
        return params

    def dists(self, params, batches):
        dists = []
        for k in range(self.signal_length):
            s_dists = []
            for j in range(self.meaning_sz):
                m_dist = []
                for i in range(self.articulator_dim):
                    mx_beta = torch.distributions.Beta(params[k][j][2 * i], params[k][j][2 * i + 1])
                    m_dist.append(mx_beta)
                s_dists.append(m_dist)
            dists.append(s_dists) # S x B (or M) x Articulator Dim
        return dists

    def decoder(self, signals):
        h0, c0 = self.get_decoder_initial_state(len(signals[0,:,0].squeeze()))
        lstm_output, _ = self.decoder_lstm(signals, (h0, c0))
        last_step = lstm_output[-1,:,:]
        y_hat = self.predictor(last_step @ self.to_softmax)

        return y_hat

    def sample_x(self, dists, batches):
        x = []
        for i in range(batches):
            x.append(torch.stack([torch.stack([torch.stack([dist.rsample() for dist in s_dists]) for s_dists in m_dists]) for m_dists in dists]))
        signals = torch.cat(x, dim=-2)
        #print(signals)
        return signals

    def forward(self, ms, batches):
        """
        To calculate loss:
        In this model alpha and beta are deterministic, therefore:
        -loop through each possible meaning
        -find alpha and beta
        -calculate likelihood of x
        """

        params = self.encoder(ms)
        dists = self.dists(params, batches)
        signals = self.sample_x(dists, batches)
        signals = signals.to(ms.device)
        y_hat = self.decoder(signals)

        return y_hat

    def train(self, epochs, print_every = 10, batches = 50):
        self.to("cuda")
        ce = torch.nn.CrossEntropyLoss().to("cuda")
        ms_in = torch.tensor(list(range(self.meaning_sz))).to(torch.int64).to("cuda")
        ms = torch.tensor(list(range(self.meaning_sz))*batches).to(torch.int64).to("cuda")

        optimizer = torch.optim.Adam(self.parameters(), lr=.001)

        for i in range(epochs):
            optimizer.zero_grad()
            y_hat = self(ms_in, batches)
            loss = ce(y_hat, ms)
            loss.backward()
            optimizer.step()

            if (i % print_every) == 0:
                print(loss)


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

