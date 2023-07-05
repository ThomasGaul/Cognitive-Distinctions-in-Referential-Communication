#pragma once

#include "CTRNN.h"

class Agent
{
    public:
        Agent (int NetworkSize, double Gain, double Range, double SpaceSize, int Dir=1);
        Agent (int NetworkSize);
        ~Agent (void) {};

        // Update Agent
        void Reset (double InitPosition);
        void SetStatus (TVector<double> &Status);
        void SetPosition (double position) { pos = position; };
        void SetDirection (double direction) { dir = direction; };
        void Sense (double other);
        void Step (double StepSize);
        void SetSensorWeight (int idx, double value) { SensorWeights(idx) = value; };

        TVector<double> &StatusVector (void);

        // Parameters
        int size, dir;
        double pos, gain, sense, range, space;
        TVector<double> SensorWeights, status;
        CTRNN NervousSystem;
};