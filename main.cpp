//*************************************************************/
//
//              Evolving Referential Communication
//               About Objects in the Environment
//
//*************************************************************/


#include <iostream>
#include <fstream>
#include <cmath>
#include <sys/stat.h>

using namespace std;

#include "random.h"
#include "VectorMatrix.h"
#include "CTRNN.h"
#include "TSearch.h"
#include "Agent.h"

// comment out the following line to only record
// #define EVOLVE

// Agent Parameters
const int N = 5;                                        // circuit size
const double WeightRange = 16.0;
const double BiasRange = 16.0;
const double SensorWeightRange = 16.0;
const double TauMin = 1.0;
const double TauMax = 30.0;
const double gain = 3.0;                                // max velocity per unit time
const double BodySize = M_PI*100 / 64.0;                // pi/64

// Task Parameters
const int Trials = 5;
const double SpaceSize = 2*M_PI*100;                    // 2 pi
const double HalfSpace = SpaceSize*0.5;
const double TransDuration = 250.0;
const double CommDuration = 300.0;
const double SearchDuration = 600.0;
const double EvalDuration = 250.0;
const double StepSize = 0.01;
const double PostSpacing = BodySize*2;                  // space between posts of the same set
const double CloseEnough = BodySize*5;                  // satisfactory distance from target
const double NoiseRange = PostSpacing;                  // variation about agent's initial position

// Recording Parameters
const double RecordStep = 0.01;
const double RobustStep = 0.01;
const int RobustTrials = 32;

// Evolution Parameters
const int PopSize = 539;
    // if using thread search: round( PopSize *(1 - Elite) ) % TRHEAD_COUNT == 0
    // THREAD_COUNT = 64
const int Gens = 20000;
const double MutVar = 0.2;
const double CrossProb = 0.0;
const double Expected = 1.1;
const double Elite = 0.05;
const double BestFitnessThreshold = 0.99;
const int SearchConstraint = 1;             // 1 is active
const int ReEvalFlag = 1;                   // 1 is active
const int DisplayToFile = 0;                // 1 is active
const int WritePopVectInterval = 50;

// EVA Search Vector
const int VectorSize = N*N + 2*N + N;

//-----------------------------------------
//            Helper Functions
//-----------------------------------------

// correct position to be periodic
inline double CircleWrapFunction (double pos)
{
    if (pos < 0.0)
        pos = pos + SpaceSize;
    if (pos >= SpaceSize)
        pos = pos - SpaceSize;
    return pos;
}

// apply variation to position 
inline double Noise(double pos, RandomState &rs)
{
    return CircleWrapFunction(pos + rs.UniformRandom(-NoiseRange, NoiseRange));
}

// calulate position of nearest post
inline double MinDistPost (double pos, TVector<double> &posts, int avail = 0)
{
    double mindist = HalfSpace;
    double dist;
    double minpost;
    if (avail < 1)
        avail = posts.Size();                           // check all posts in vector if no limit specified
    for (int i = 1; i <= avail; ++i) {
        dist = fabs(pos - posts(i));
        if (dist > HalfSpace)
            dist = SpaceSize - dist;
        if (dist <= mindist) {                          // update nearest post
            mindist = dist;
            minpost = posts(i);
        }
    }
    return minpost;
}

// apply genotype to an agent's CTRNN
inline void GenPhenMapping (TVector<double> &genotype, Agent &a)
{
    // map genotype to agents
    int k = 1;

    // time constants
    for (int i = 1; i <= N; ++i) {
        a.NervousSystem.SetNeuronTimeConstant(i, MapSearchParameter(genotype(k), TauMin, TauMax));
        ++k;
    }
    
    // biases
    for (int i = 1; i <= N; ++i) {
        a.NervousSystem.SetNeuronBias(i, MapSearchParameter(genotype(k), -BiasRange, BiasRange));
        ++k;
    }

    // connection weights
    for (int i = 1; i <= N; ++i) {
        for (int j = 1; j <= N; ++j) {
            a.NervousSystem.SetConnectionWeight(i,j, MapSearchParameter(genotype(k), -WeightRange, WeightRange));
            ++k;
        }
    }

    // sensor weights
    for (int i = 1; i <= N; ++i) {
        a.SetSensorWeight(i, MapSearchParameter(genotype(k), -SensorWeightRange, SensorWeightRange));
        ++k;
    }
}

// convert command line argument (char) to int
inline int ArgtoInt (char* arg)
{
    int x;
    sscanf(arg,"%d",&x);
    return x;
}

// check for file existence
inline bool exists (const std::string& name) {
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}

// Write paramters to file
void ParametersToFile (int evol=0) // 1 for evolution parameters
{
    // general parameters
    if (evol && !exists("evol_parameters.dat")) {
        ofstream para;
        para.open("evol_parameters.dat");
        
        para << "N,BodySize,SpaceSize,TransDuration,"
            << "CommDuration,SearchDuration,StepSize,Population,Gen\n";
        
        para << N << "," << BodySize << "," << SpaceSize << ","
            << TransDuration << "," << CommDuration << "," << SearchDuration
            << StepSize << "," << PopSize << "," << Gens;
    
        para.close();
    }
    else if (!evol && !exists("rec_parameters.dat")) {
        ofstream para;
        para.open("rec_parameters.dat");
    
        para << "N,BodySize,SpaceSize,TransDuration,"
            << "CommDuration,SearchDuration,StepSize\n";
        
        para << N << "," << BodySize << "," << SpaceSize << ","
            << TransDuration << "," << CommDuration << "," << SearchDuration
            << "," << RecordStep;

        para.close();
    }

    // phenotype
    if (!evol && !exists("phenotype.dat")) {

        ifstream genefile;
        genefile.open("genefile.dat");
        TVector<double> genotype;
        genotype.SetBounds(1,VectorSize);
        genefile >> genotype;
        genefile.close();

        ofstream phenotype;
        phenotype.open("phenotype.dat");
        Agent best (N);
        GenPhenMapping(genotype,best);
        phenotype << best.NervousSystem;
        phenotype << "\n" << best.SensorWeights;
        phenotype.close();
    }
}

//-----------------------------------------
//          Analysis Functions
//-----------------------------------------
void Record (TVector<double> &genotype, RandomState &rs, int permu);
void Robust (TVector<double> &genotype, RandomState &rs);

//======================================================
//                  Fitness Function
//======================================================

double FitnessFunction (TVector<double> &genotype, RandomState &rs)
{   
    // define agents
    Agent Sender (N, gain, BodySize, SpaceSize, 1);
    Agent Receiver (N, gain, BodySize, SpaceSize, -1);

    // map genotype to CTRNN parameters
    GenPhenMapping(genotype, Sender);
    GenPhenMapping(genotype, Receiver);

    // initialize lowest trial average
    double lowfit = 1.1;

    for (int trial = 1; trial <= Trials; ++trial) {
        
        // permutations
        int perm = 4;                       // 2 target * 2 pos
        TVector<double> perf;
        perf.SetBounds(1,perm);
        int assign = 1;                     // target tag

        for (int p = 1; p <= perm; ++p) {
            
            // Transient Phase
            if (p > 2)
                assign = 2;                                                 // update target tag
            TVector<double> posts;
            posts.SetBounds(1,3);
            posts.FillContents(0.0);
            posts(1) = HalfSpace;
            if (assign == 2)
                posts(2) = posts(1) + PostSpacing;

            Sender.Reset(Noise(0.0, rs));                                   // initialize sender
            for (double t = 0.0; t < TransDuration; t += StepSize) {

                // sense
                Sender.Sense(MinDistPost(Sender.pos, posts, assign));
                
                // update and move
                Sender.Step(StepSize);
            }

            // failure if sender does not decouple from post
            if (fabs(Sender.pos - MinDistPost(Sender.pos, posts, assign)) < CloseEnough) {
                perf(p) = 0.0; continue;
            }
            

            // Communication Phase
            Receiver.Reset(Noise(Sender.pos + HalfSpace, rs));              // initial receiver
            for (double t = 0.0; t < CommDuration; t += StepSize) {

                // sense
                Sender.Sense(Receiver.pos);
                Receiver.Sense(Sender.pos);

                // update and move
                Sender.Step(StepSize);
                Receiver.Step(StepSize);
            }
            
            // failure if agents do not decouple
            double dist = fabs(Sender.pos - Receiver.pos);
            if (dist > HalfSpace)
                dist = SpaceSize - dist;
            if (dist < CloseEnough) {
                perf(p) = 0.0; continue;
            }

            // Search Phase
            int assign_pos = (p % 2) + 1;                                   // alternate position of target
            posts(1) = (assign_pos * SpaceSize) / 3.0;
            posts(assign+1) = (((assign_pos%2)+1) * SpaceSize) / 3.0;       // alt. post at other position
            for (int i = 2; i <= assign; ++i)                               // add remaining posts
                posts(i) = posts(i-1) + PostSpacing;
            for (int i = assign+2; i <= 3; ++i)
                posts(i) = posts(i-1) + PostSpacing;

            double permdist = 0.0, permtime = 0.0;                          // fitness parameters for permutation
            int contact = 0, confuse = 0;                                   // flags to check if receiver "confuses" the posts  
            
            Receiver.SetPosition(Noise(0.0, rs));                           // reposition receiver (but do not reset)
            for (double t = 0.0; t < SearchDuration; t += StepSize) {

                // sense
                Receiver.Sense(MinDistPost(Receiver.pos, posts));

                // update and move
                Receiver.Step(StepSize);
                
                double dist = fabs(Receiver.pos - MinDistPost(Receiver.pos,posts,assign));
                if (t > SearchDuration - EvalDuration) {                                      // evaluate distance to target
                    if (dist < CloseEnough)
                        dist = CloseEnough;
                    permdist += dist;
                    ++permtime;
                }
                else if (contact == 0 && dist < CloseEnough)
                    ++contact;                                                              // initial contact with post
                else if (contact != 0 && Receiver.sense > 0 && dist > CloseEnough) {
                    ++confuse; break;                                                       // receiver contacts alt. post
                }
            }
            
            // evaluate performance
            if (confuse != 0) {
                perf(p) = 0.0; continue;
            }
            perf(p) = 1 - ( ((permdist / permtime) - CloseEnough) / (0.3*SpaceSize - CloseEnough) );
            if (perf(p) < 0)
                perf(p) = 0.0;

        }
        
        // evaluate at average across permutations
        double fit = 0.0;
        for (int i = 1; i <= perm; ++i)
            fit += perf(i);
        fit /= (double)perm;

        // compare to lowest per trail
        if (fit == 0) {
            lowfit = 0.0; break;
        }
        else if (fit < lowfit) {
            lowfit = fit; continue;
        }
    }

    return lowfit;
}

//======================================================
//                  Evolution Functions
//======================================================
//---------------------------------------
//    Statistics Display Function
//---------------------------------------
void StatDisplay (int Gen, double BestPerf, double AvgPerf, double PerfVar)
{
    if (DisplayToFile == 1) {
        ofstream file;
        file.open("update.dat", fstream::app);
        file << "Gen: " << Gen
        << "\tBest: " << BestPerf
        << "\tAvg: " << AvgPerf
        << "\tVar: " << PerfVar
        << "\n";
        file.close();
    }
    else {
        std::cout << "Gen: " << Gen
        << "\tBest: " << BestPerf
        << "\tAvg: " << AvgPerf
        << "\tVar: " << PerfVar
        << "\n";
    }
}

//---------------------------------------
//  Write Population Vector Function
//---------------------------------------
void WritePopStat (TVector<double> &perf, int Gen)
{
    ofstream file;
    file.open("evol.dat", fstream::app);
    file << Gen << " " << perf << "\n";
    file.close();
}

//---------------------------------------
//    Best Individual Found Function
//---------------------------------------
void BestFound (int Gen, TVector<double> &genotype, double BestPerf)
{
    ofstream best, bestgene;
    best.open("evolbest.dat", fstream::app);
    bestgene.open("evolgene.dat", fstream::app);
    best << Gen << " " << BestPerf << "\n";
    bestgene << Gen << " " << genotype << "\n";
    best.close(); bestgene.close();
}

//---------------------------------------
//        Terminiation Function
//---------------------------------------
int TerminationFunction (int Gen, double BestPerf, double AvgPerf, double PerfVar)
{
    if (BestPerf >= BestFitnessThreshold) return 1;
    else if (Gen == 10000 && BestPerf < 0.85) return 1;
    else return 0;
}

//---------------------------------------
//      Results Display Function
//---------------------------------------
void Results (TSearch &s)
{
    // write best agent genotype to file
    ofstream BestGenotype;
    BestGenotype.open("genefile.dat");
    TVector<double> genotype = s.BestIndividual();
    BestGenotype << genotype;
    BestGenotype.close();

    // write best agent phenotype to file
    ofstream BestPhenotype;
    BestPhenotype.open("phenotype.dat");
    Agent best (N);
    GenPhenMapping(genotype,best);
    BestPhenotype << best.NervousSystem;
    BestPhenotype << " " << best.SensorWeights;
    BestPhenotype.close();

    // write population fitness vector to file
    ofstream FitnessVector;
    FitnessVector.open("FinalFitnessVector.dat");
    for (int i = 1; i <= PopSize; ++i) 
        FitnessVector << s.Performance(i) << " ";
    FitnessVector.close();
}

//---------------------------------------
//    Vector Initialization Function
//---------------------------------------
// for starting searches from a given genotype
void InitializeVector (TVector<double> &v, RandomState &rs)
{
    // check for vector size compatibility
    if (v.Size() != VectorSize) {
        std::cerr << "search vector sizes do not match.\n";
        return;
    }

    // get genotype
    ifstream seed;
    seed.open("../../initial_seed.dat");
    seed >> v;
    seed.close();
    
    // apply noise to vector
    for (int i = 1; i <= VectorSize; ++i)
        v(i) += rs.UniformRandom(-0.1,0.1);
    
    return;
}

//==================================================
//                  Main Function
//==================================================
int main(int argc, char *argv[])
{
    //---------------------------------------------
    //                  Evolution
    //---------------------------------------------
    #ifdef EVOLVE

    // write parameters to file
    ParametersToFile(1);

    // generate and write seed to file
    long wseed = static_cast<long>(time(NULL));
    if (argc == 2)
        wseed *= ArgtoInt(argv[1]);     // seed by input in case of parallel instances
    else if (argc > 2) {
        std::cerr << "too many arguments: " << argc << "/2\n";
        return 1;
    }

    // write random seed to file
    ofstream seedfile;
    seedfile.open("seed.dat");
    seedfile << wseed;
    seedfile.close();

    // configure evolution
    TSearch s(VectorSize);

    s.SetRandomSeed(wseed);
    s.SetPopulationSize(PopSize);
    s.SetMaxGenerations(Gens);
    s.SetSelectionMode(RANK_BASED);
    s.SetReproductionMode(GENETIC_ALGORITHM);
    s.SetCrossoverMode(UNIFORM);
    s.SetCrossoverProbability(CrossProb);
    s.SetMutationVariance(MutVar);
    s.SetMaxExpectedOffspring(Expected);
    s.SetElitistFraction(Elite);
    s.SetSearchConstraint(SearchConstraint);
    s.SetReEvaluationFlag(ReEvalFlag);
    s.SetCheckpointInterval(500);
    s.SetPopulationVectorInterval(WritePopVectInterval);
    
    // clear/create files before evolution
    ofstream best, evol;
    best.open("evolbest.dat");
        best << "Generation Best\n"; // header
    evol.open("evol.dat");
    best.close(); evol.close();

    // Function Pointers
    s.SetSearchTerminationFunction(TerminationFunction);
    s.SetEvaluationFunction(FitnessFunction);
    s.SetWritePopulationVectorFunction(WritePopStat);
    s.SetPopulationStatisticsDisplayFunction(StatDisplay);
    s.SetBestActionFunction(BestFound);
    s.SetSearchResultsDisplayFunction(Results);
    
    // Intialize from given vector
    // s.InitializeSearch();
    // for (int i = 1; i <= s.PopulationSize(); ++i)
    //     InitializeVector(s.Individual(i),s.IndividualRandomState(i));
    
    // Execute Search
    s.ExecuteSearch();
    // s.ResumeSearch();

    if (s.BestPerformance() < 0.90)
        std::cout << "failed to acheive fitness > 0.90: " << s.BestPerformance() << "\n";
    else
        std::cout << "acheived fitness > 0.90: " << s.BestPerformance() << "\n";
    
    return 0;
   
    #endif
    
    #ifndef EVOLVE
    
    // check for arguments
    if (argc != 2) return 1;
    
    // read in seed (main should be called from specific population directory)
    long rseed;
    TVector<long> temp; temp.SetSize(1);
    ifstream readseed;
    readseed.open("seed.dat");
    readseed >> temp;
    rseed = temp[1];
    readseed.close();

    RandomState rs; rs.SetRandomSeed(rseed);
    
    //---------------------------------------
    //          Robustness Function
    //---------------------------------------
    if (*argv[1] == 'r') {
            
            // get genotype
            ifstream genefile;
            genefile.open("genefile.dat");
            TVector<double> genotype;
            genotype.SetSize(VectorSize);
            genefile >> genotype;
            genefile.close();

            Robust(genotype, rs);
    }
    
    //----------------------------
    //          Recording
    //----------------------------
    else {

        ParametersToFile();

        // get genotype
        ifstream genefile;
        genefile.open("genefile.dat");
        TVector<double> genotype;
        genotype.SetSize(VectorSize);
        genefile >> genotype;
        genefile.close();
            
        // record for all permutations
        if (*argv[1] == 'a') {
            for (int p = 1; p <= 4; ++p)
                Record(genotype, rs, p);
        }
        // record for a given permutation
        else
            Record (genotype, rs, ArgtoInt(argv[1]));
    }
    #endif

    return 0;
}

//----------------------------------------------------
//              functions for analysis
//----------------------------------------------------
void Robust (TVector<double> &genotype, RandomState &rs)
{
    // open files
    string step;
    if (RobustStep == 0.01)
        step = "01";
    else if (RobustStep == 0.001)
        step = "001";
    else if (RobustStep == 0.0001)
        step = "0001";

    string pf_file = "robust/RPF-" + step + ".dat";
    ofstream pf; pf.open(pf_file);

    Agent Sender (N, gain, BodySize, SpaceSize, 1);
    Agent Receiver (N, gain, BodySize, SpaceSize, -1);

    GenPhenMapping(genotype, Sender);
    GenPhenMapping(genotype, Receiver);
    
    // vectors for lowest and average performance per trial
    TVector<double> robust; TVector<double> avg;
    robust.SetBounds(1,RobustTrials);
    avg.SetBounds(1,RobustTrials);

    for (int trial = 1; trial <= RobustTrials; ++trial) {
        
        int perm = 4;
        TVector<double> perf;
        perf.SetBounds(1,perm);
        int assign = 1; 
        
        for (int p = 1; p <= perm; ++p) {
            
            // Transient Phase
            if (p > 2)
                assign = 2;
            TVector<double> posts;
            posts.SetBounds(1,3);
            posts.FillContents(0.0);
            posts(1) = HalfSpace;
            if (assign == 2)
                posts(2) = posts(1) + PostSpacing;

            Sender.Reset(Noise(0.0, rs));
            for (double t = 0.0; t < TransDuration; t += RobustStep) {

                Sender.Sense(MinDistPost(Sender.pos, posts, assign));
                
                Sender.Step(RobustStep);
            }

            if (fabs(Sender.pos - MinDistPost(Sender.pos, posts, assign)) < BodySize) {
                perf(p) = 0.0; continue;
            }
            
            // Communication Phase
            Receiver.Reset(Noise(Sender.pos + HalfSpace, rs));
            for (double t = 0.0; t < CommDuration; t += RobustStep) {

                Sender.Sense(Receiver.pos);
                Receiver.Sense(Sender.pos);

                Sender.Step(RobustStep);
                Receiver.Step(RobustStep);
            }
            
            double dist = fabs(Sender.pos - Receiver.pos);
            if (dist > HalfSpace)
                dist = SpaceSize - dist;
            if (dist < BodySize) {
                perf(p) = 0.0; continue;
            }

            // Search Phase
            int assign_pos = (p % 2) + 1;
            posts(1) = (assign_pos * SpaceSize) / 3.0;
            posts(assign+1) = (((assign_pos%2)+1) * SpaceSize) / 3.0;
            for (int i = 2; i <= assign; ++i)
                posts(i) = posts(i-1) + PostSpacing;
            for (int i = assign+2; i <= 3; ++i)
                posts(i) = posts(i-1) + PostSpacing;

            double permdist = 0.0, permtime = 0.0;
            int contact = 0, confuse = 0;
            
            Receiver.SetPosition(Noise(0.0, rs));
            for (double t = 0.0; t < SearchDuration; t += RobustStep) {

                Receiver.Sense(MinDistPost(Receiver.pos, posts));

                Receiver.Step(RobustStep);
                
                double dist = fabs(Receiver.pos - MinDistPost(Receiver.pos,posts,assign));
                if (t > SearchDuration - EvalDuration) {
                    if (dist < CloseEnough)
                        dist = CloseEnough;
                    permdist += dist;
                    ++permtime;
                }
                else if (contact == 0 && dist < CloseEnough)
                    ++contact;
                else if (contact != 0 && Receiver.sense > 0 && dist > CloseEnough) {
                    ++confuse; break;
                }
            }
            if (confuse != 0) {
                perf(p) = 0.0; continue;
            }
            perf(p) = 1 - ((permdist / permtime) - CloseEnough) / (0.33*SpaceSize - CloseEnough);
            if (perf(p) < 0)
                perf(p) = 0;
        }

        // evaluate at lowest performance and average per permutation
        double fit = 1.1, avgfit = 0.0;
        for (int i = 1; i <= perm; ++i) {
            avgfit += perf(i);
            if (perf(i) == 0)
                fit = 0;
            else if (perf(i) < fit)
                fit = perf(i);
            else continue;
        }
        avgfit /= (double)perm;
        pf  << perf << "\n";
        robust(trial) = fit;
        avg(trial) = avgfit;
    }
    
    // evaluate at lowest performance and average per trial
    double fit = 1.1, avgfit = 0.0;
    for (int i = 1; i <= RobustTrials; ++i) {
        avgfit += avg(i);
        if (robust(i) < fit)
            fit = robust(i);
        else continue;
    }
    avgfit /= (double)RobustTrials;

    pf << fit << " " << avgfit;
    pf.close();
    std::cout << fit << " " << avgfit << " ";

    return;
}

void Record (TVector<double> &genotype, RandomState &rs, int permu=1)   
{
    // open files (sender, receiver, posts, performance)
    ofstream s, r, pt, pf;
    string s_file = "record/S-p" + to_string(permu) + ".dat";
    string r_file = "record/R-p" + to_string(permu) + ".dat";
    string pt_file = "record/P-p" + to_string(permu) + ".dat";
    string pf_file = "record/PF.dat";
    string ph_file = "phenotype.dat";
    s.open(s_file); r.open(r_file); pt.open(pt_file); pf.open(pf_file);

    // write labels
    string labels = "Position Sensor Motor1 Motor2";
    for (int lab = 3; lab <= N; ++lab)
        labels += " Inter" + to_string(lab - 2);
    s << labels << "\n";
    r << labels << "\n";

    s << "# Phase 1";
    r << "# Phase 2";

    Agent Sender (N, gain, BodySize, SpaceSize, 1);
    Agent Receiver (N, gain, BodySize, SpaceSize, -1);

    GenPhenMapping(genotype, Sender);
    GenPhenMapping(genotype, Receiver);

    int perm = 4;
    TVector<double> perf;
    perf.SetBounds(1,perm);
    int assign = 1;

    for (int p = 1; p <= perm; ++p) {

        // Transient Phase
        if (p > 2)
            assign = 2;
        TVector<double> posts;
        posts.SetBounds(1,3);
        posts.FillContents(0.0);
        posts(1) = HalfSpace;
        if (assign == 2)
            posts(2) = posts(1) + PostSpacing;

        // write post positions to file
        if (p == permu)
            pt << posts << "\n";

        Sender.Reset(0.0);
        for (double t = 0.0; t < TransDuration; t += RecordStep) {

            Sender.Sense(MinDistPost(Sender.pos, posts, assign));
            
            Sender.Step(RecordStep);

            if (p == permu) {
                s << "\n";
                s << Sender.StatusVector();
            }
        }

        if (fabs(Sender.pos - MinDistPost(Sender.pos, posts, assign)) < BodySize)
            perf(p) = 0.0;
        
        // write header
        if (p == permu)
            s << "\n\n# Phase 2\n";

        // Communication Phase
        Receiver.Reset(CircleWrapFunction(Sender.pos + HalfSpace));
        for (double t = 0.0; t < CommDuration; t += RecordStep) {

            Sender.Sense(Receiver.pos);
            Receiver.Sense(Sender.pos);

            Sender.Step(RecordStep);
            Receiver.Step(RecordStep);

            if (p == permu) {
                s << "\n";
                r << "\n";
                s << Sender.StatusVector();
                r << Receiver.StatusVector();
            }
        }
        
        double dist = fabs(Sender.pos - Receiver.pos);
        if (dist > HalfSpace)
            dist = SpaceSize - dist;
        if (dist < CloseEnough)
            perf(p) = 0.0;

        // Search Phase
        int assign_pos = (p % 2) + 1;
        posts(1) = (assign_pos * SpaceSize) / 3.0;
        posts(assign+1) = (((assign_pos%2)+1) * SpaceSize) / 3.0;
        for (int i = 2; i <= assign; ++i)
            posts(i) = posts(i-1) + PostSpacing;
        for (int i = assign+2; i <= 3; ++i)
            posts(i) = posts(i-1) + PostSpacing;

        // write header and post positions
        if (p == permu) {
            pt << posts;
            r << "\n\n# Phase 3\n";
        }

        double permdist = 0.0, permtime = 0.0;
        int contact = 0, confuse = 0;

        Receiver.SetPosition(0.0);
        for (double t = 0.0; t < SearchDuration; t += RecordStep) {

            Receiver.Sense(MinDistPost(Receiver.pos, posts));

            Receiver.Step(RecordStep);

            if (p == permu) {
                r << "\n";
                r << Receiver.StatusVector();
            }

            double dist = fabs(Receiver.pos - MinDistPost(Receiver.pos,posts,assign));
            if (t > SearchDuration - EvalDuration) {
                if (dist < CloseEnough)
                    dist = CloseEnough;
                permdist += dist;
                ++permtime;
            }
            else if (contact == 0 && dist < CloseEnough)
                ++contact;
            else if (contact != 0 && Receiver.sense > 0 && dist > CloseEnough)
                ++confuse;
        }
        if (confuse != 0) {
            perf(p) = 0.0; continue;
        }
        perf(p) = 1 - ((permdist / permtime) - CloseEnough) / (0.33*SpaceSize - CloseEnough);
        if (perf(p) < 0)
            perf(p) = 0.0;
    }

    pf << perf << "\n";

    // evaluate at average
    double fit = 0.0;
    for (int i = 1; i <= perm; ++i)
        fit += perf(i);
    fit /= (double)perm;
 
    pf << fit;

    s.close(); r.close(); pt.close(); pf.close();
    return;
}
