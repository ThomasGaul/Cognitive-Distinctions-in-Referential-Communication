 # Cognitive Distinctions in Referential Communciation

This is a minimal model of referential communication. This repository contains the code for the evolution and analysis of agents in a referential communication task.

The environment is a one-dimensional ring with posts that the agents can interact with. The agents are 5-neuron CTRNNs with a body and a single continuous sensor. The task is organised into three phases:

1. *Transient Phase*. The sender interacts with the target to determine its identity (whether it is 1 or 2 posts).
2. *Communication Phase*. The sender interacts with the receiver to communicate the identity of the target.
3. *Search Phase*. The receiver finds and stays near the target, ignoring the other post(s).

Any questions or concerns pertaining to the model or the code can be sent to my email: tgaul@iu.edu.
