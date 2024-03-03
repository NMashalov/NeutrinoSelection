# Neutrino Selection

### Abstract

Baikal-GVD is a large (∼ 1 km3) underwater neutrino telescope located in Lake Baikal, Russia. This work presents a neural network for separatingeventscausedbyextensive air showers (EAS) and neutrinos. By choosing appropriate classification threshold, we preserve 50% of neutrino-induced events,while EAS-inducedevents are suppressed by a factor of 10-6. A method for estimating the neutrino flux with minimal error based on neural network predictions has alsobeendeveloped. The developed neural network employ the causal structure of events and surpass the precision of standard algorithmic approaches.

---


<div align="center">
 <img width="460" height="300" src=assets/Neutrino.png>
 <table>
    <tr>
        <th><a href=https://indico.jinr.ru/event/3792/contributions/23465/attachments/17254/29376/AYSS_Matseiko_poster.pdf> 🖼️Poster</a> </th>
        <th> <a>📄Paper</a> </th>
    </tr>
 </table>
</div>

---

The repository contains framework for  data analysis in [Baikal-GVD experiment](https://baikalgvd.jinr.ru/). 
Neural networks solve problem of separation neutrino-induced events from the background of [EAS's]().

Repository provides code for Physical-informed neural nets:

---

### Functionality 

- Linear and Convolution NN with fixed data
- Recurrent Neural Networks for work with sequential data
- HyperNetworks for marginal optimization 

## Getting started

### Installation

Preliminaries: I recommend using Linux distributions 


Clone repository using [git](https://git-scm.com/)  
```
git clone 
```

[Poetry](https://python-poetry.org/) will install proper environment for your start 

```python

poetry init
```

### Command Line Interface

After installation project can be started solely from command: 

```
    neutrino_selection start <WRITE YOUR h5 data> --architecture <CHOOSE ARCHITECTURE>
```

Possible options are:
- NN 
- RNN
- HYPER 

### Python 

Minimal example

```python
from neutrino_selection.net import RnnHyperNetwork
from neutrino_selection.data import H5Dataset

dataset = H5Dataset('<YOUR DATA>')

net = RnnHyperNetwork(device='cuda')

net.train(dataset)
```

Proceed to tutorials for accustomation with framework

## Contact

Contact me via opening issue on github or sending email matseiko.av@phystech.edu.

Use telegram for research collaboration `@AlbertMac280`