# *This repository has been migrated to [GitLab](https://gitlab.com/tgaul/referential-communication).*
---
## Cognitive Distinctions in a Model of Referential Communciation

This is a minimal model of referential communication used to explore the concept of cognitive distinctions. This repository contains the code for the evolution and analysis of agents in a referential communication task. The work has been [published in *Artificial Life*](https://doi.org/10.1162/artl_a_00475). A preprint is freely available on [my website](https://tgaul.gitlab.io/publications/2025-distinctions).

The environment is a one-dimensional ring with 'post-sets' that agents can interact with. The agents are 5-neuron CTRNNs with a single continuous sensor. The task is organised into three phases:

1. *Phase 1*. The sender interacts with the target to differentiate whether it has one or two posts.
2. *Phase 2*. The sender interacts with the receiver to prepare the latter to find the correct post-set.
3. *Phase 3*. The receiver finds and stays near the target post-set, ignoring the distraction post-set.

For the rest of the data, please contact me at [gaul@uchc.edu](mailto:gaul@uchc.edu). Please also reach out for any questions or concerns. 

### Software Used
| Software  | Version   |
| --------- | --------- |
| gcc       | 12.2.0    |
| python     | 3.12.4    |
| Wolfram   | 14.1.0    |
| OS        | Debian 12.2.0-14 (*Bookworm*) |

---

If you want to cite the paper, use:
```bib
@article{Gaul.Izquierdo2025distinctions,
    author = {Gaul, Thomas M and Izquierdo, Eduardo J},
    title = {Cognitive distinctions as a language for cognitive science},
    subtitle = {comparing methods of description in a model of referential communication},
    year = {2025},
    month = {09},
    journal = {Artificial Life},
    volume = {31},
    number = {3},
    pages = {345--367},
    publisher = {MIT Press},
    doi = {10.1162/artl_a_00475}
}
```
