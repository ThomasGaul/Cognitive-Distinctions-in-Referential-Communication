#include "Pair.h"

agent::agent (int N, double Gain, double BodySize, double SpaceSize, int Dir)
{
    size = N;
    SensorWeights.SetBounds(1, N);
    SensorWeights.FillContents(0.0);
    gain = Gain;
    range = BodySize;       // contact range
    dir = Dir;
    space = SpaceSize;
    NervousSystem.SetCircuitSize(N);
    status.SetBounds(1, N+2);
    return;
}


void agent::Reset (double InitPosition)
{
    sense = 0.0;
    pos = InitPosition;
    for (int i = 1; i <= size; ++i) {
        NervousSystem.SetNeuronState(i,0.0);
    }
    return;
}

// initial agent from a specified state
void agent::SetStatus (TVector<double> &Status)
{
    sense = 0.0;
    pos = Status(1);
    for (int i = 1; i <= size; ++i) {
        NervousSystem.SetNeuronOutput(i,Status(i+2));
    }
    status = StatusVector();
    return;
}

void agent::Sense(double other)
{
    // calculate distance to object (post or agent)
    double dist = fabs(pos - other);
    if (dist > 0.5*space) { dist = space - dist; }

    if (dist > range) { sense = 0; return; }
    else {
        dist /= range;                          // normalize to [0,1]
        sense = 1 / (1 + exp(10*dist - 5));     // calculate sense value
        return;
    }
}


void agent::Step(double StepSize)
{
    // apply sensor to cuircuit
    for (int i = 1; i <= size; ++i) {
        NervousSystem.SetNeuronExternalInput(i, SensorWeights[i] * sense);
    }

    // update nervous system
    NervousSystem.EulerStep(StepSize);

    // update position
    pos += StepSize * gain * dir * (NervousSystem.NeuronOutput(1) - NervousSystem.NeuronOutput(2));

    // wrap around circle
    if (pos >= space) {
        pos -= space;
    }
    if (pos < 0.0) {
        pos += space;
    }
    return;
}

// vector for writing basic information to file
TVector<double> &agent::StatusVector (void)
{
    status[1] = pos; status[2] = sense;
    for (int i = 1; i <= size; ++i) {
        status[i+2] = NervousSystem.NeuronOutput(i);
    }
    return status;
}