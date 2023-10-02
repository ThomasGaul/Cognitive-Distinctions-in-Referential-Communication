 # Cognitive Distinctions in Referential Communciation

This is a minimal model of referential communication. This repository contains the code for the evolution and analysis (behaviour, parameter space, dynamics) of agents in a referential communication task. As of October 2023, the work has been described in a article submitted to the Artifical Life Journal.

The environment is a one-dimensional ring with posts that the agents can interact with. The agents are 5-neuron CTRNNs with a body and a single continuous sensor. The task is organised into three phases:

1. *Transient Phase*. The sender interacts with the target to determine its identity (whether it is 1 or 2 posts).
2. *Communication Phase*. The sender interacts with the receiver to communicate the identity of the target.
3. *Search Phase*. The receiver finds and stays near the target, ignoring the other post(s).

For data, please contact me at tgaul@iu.edu. Please also reach out for any questions or concerns pertaining to the model (or anything else).
