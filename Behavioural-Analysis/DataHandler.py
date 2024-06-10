import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3D
from copy import copy

# Class for visualizing a given population

# files and directories must be formatted correctly

class Data:
#-----------------------------------------
#   Initilization and access functions
#-----------------------------------------
    def __init__ (self, base, maxP, popidx, perm=[0],batchidx=0):
        
        # use absolute path (more secure file reading)
        self.SetBaseDirectory(base)
        
        # Directory to specific agent
        if maxP == 2:
            self.Directory = "./EvoP" + str(maxP) + "/Pop-" + str(popidx) + "/record"
        if maxP == 3:
            self.Directory = "./AllData/EvoP3_" + str(batchidx) + "/Pop-" + str(popidx) + "/record"
        os.chdir(self.Directory)
        
        # parameters of recorded data
        os.chdir("../")
        self.p = pd.read_table("rec_parameters.dat", delimiter=",")
        self.p_evo = pd.read_table("evol_parameters.dat", delimiter=",")
        os.chdir(self.BaseDirectory)
        os.chdir(self.Directory)

        # set permutation indices
        self.P = maxP
        if (self.P == 2):
            self.MaxPosts = 4
        elif (self.P == 3):
            self.MaxPosts = 12
        self.Permutations = []
        if (perm != None and perm[0] == 0):
            self.SetPermutationIndices(np.arange(1,self.MaxPosts+1,1),init=True)
        elif (perm != None):
            self.SetPermutationIndices(perm, init=True)

        self.DataInitialization()

        # plot data for position
        self.PosLines_2D = [np.full((len(self.Permutations),2), None),np.full((len(self.Permutations),2), None)]
        self.PosLines_3D = [np.full((len(self.Permutations),2), None),np.full((len(self.Permutations),2), None)]
        self.PostLines_2D = np.full((len(self.Permutations),2), None)
        self.PostLines_3D = np.full((len(self.Permutations),2), None)
            # agent -> permutation -> phase
        
        # plot data for neuron activation
        phase_act = [np.full(5,None),np.full(5,None)]
        self.ActLines = [
            np.full((len(self.Permutations),2,5), phase_act),
            np.full((len(self.Permutations),2,5), phase_act)
            ]
            # agent -> permutation -> phase -> neuron
        
        # plot data for sensors
        self.SensLines = [np.full((len(self.Permutations),2), None),np.full((len(self.Permutations),2), None)]

        # plot data for contact points
        self.ContactPoints2D = np.full((len(self.Permutations),3),None)
        self.ContactPoints3D = np.full((len(self.Permutations),3),None)
            # permutation -> phase

        os.chdir(self.BaseDirectory)
        return
    
    def SetBaseDirectory(self,dir):
        self.BaseDirectory = dir
        os.chdir(dir)
        return

    # permutations must be list or array
    def SetPermutationIndices (self,permutations,init=False):
        
        # clear set
        self.Permutations = []
        # convert given set to string
        for i in permutations:
            self.Permutations.append(str(i))
            continue
        # reload data unless called by object initialiation
        if (init == True): return
        self.DataInitialization()
        return
    
    # converts given permutation to appropriate index
    def PermutationIndex (self,perm):
        for i in range(len(self.Permutations)):
            if (str(perm) == self.Permutations[i]):
                return i
        else:
            raise KeyError("permutation " + str(perm) + " does not exist or has not been loaded")
    
    def NeuronIndex (self,n):
        return self.Sender[0][0].columns[n+1] 

    def ShowFitness(self):
        print(self.FitnessVector)
        print(self.TotalFitness)
        return

    def DataInitialization (self):
        
        # basic fitness data
        if len(self.Permutations) == self.MaxPosts:
            PF_file = "PF.dat"
            self.FitnessVector = np.loadtxt(PF_file, dtype=float, max_rows=1)
            self.TotalFitness = np.loadtxt(PF_file, dtype=float, skiprows=1, max_rows=1)

        # list containers for data of multiple permutations
        self.Posts = []
        self.Sender = []
        self.Receiver = []
        self.Time = []

        # arrays for x-axis in time series plots
        self.Time.append(
            np.linspace(0, self.p["Phase1Duration"][0], num=(int(self.p["Phase1Duration"][0]/self.p["StepSize"][0]))+1 )
        )
        self.Time.append(
            np.linspace(0, self.p["Phase2Duration"][0], num=(int(self.p["Phase2Duration"][0]/self.p["StepSize"][0]))+1 )
        )
        self.Time.append(
            np.linspace(0, self.p["Phase3Duration"][0], num=(int(self.p["Phase3Duration"][0]/self.p["StepSize"][0]))+1 )
        )
        
        for perm in self.Permutations:
            S_file = "S-p" + perm + ".dat"
            R_file = "R-p" + perm + ".dat"
            P_file = "P-p" + perm + ".dat"

            # labels to apply to each dataframe
            labels = pd.read_table(S_file, delimiter=" ", header=0, nrows=1).columns

            # load posts per permutation
            self.Posts.append(pd.read_table(P_file,delimiter=" ",header=None, nrows=2))

            TempSender = []
            TempReceiver = []
            
            # Phase 1 Time Series
            TempSender.append(
                pd.read_table(S_file,delimiter=" ",names=labels,
                    nrows = (int(self.p["Phase1Duration"][0]/self.p["StepSize"][0])) + 1,
                    skiprows = 2
                )
            )

            # Phase 2 Time Series
            TempSender.append(
                pd.read_table(S_file,delimiter=" ",names=labels,
                    nrows = int(self.p["Phase2Duration"][0]/self.p["StepSize"][0]) + 1,
                    skiprows = 5 + int(self.p["Phase1Duration"][0]/self.p["StepSize"][0])
                )
            )
            TempReceiver.append(
                pd.read_table(R_file,delimiter=" ",names=labels,
                    nrows = (int(self.p["Phase2Duration"][0]/self.p["StepSize"][0])) + 1,
                    skiprows = 2
                )
            )

            # Phase 3 Time Series
            TempReceiver.append(
                pd.read_table(R_file,delimiter=" ",names=labels,
                    nrows = int(self.p["Phase3Duration"][0]/self.p["StepSize"][0]) + 1,
                    skiprows = 5 + int(self.p["Phase2Duration"][0]/self.p["StepSize"][0])
                )
            )

            self.Sender.append(TempSender)
            self.Receiver.append(TempReceiver)

            os.chdir(self.BaseDirectory); os.chdir(self.Directory)
            continue


        # Best Fitness Time Series
        os.chdir(self.BaseDirectory); os.chdir(self.Directory)
        self.EvolBest = pd.read_table("../evolbest.dat",delimiter=" ",header=0)

        # Population Time Series
        self.EvolPop = pd.read_table("../evol.dat",delimiter=" ",header=None)
        self.EvolPop.rename(columns={0:"Generation"},inplace=True)
        self.EvolPop.drop_duplicates(subset="Generation",keep="last",inplace=True)
        self.EvolPop.reset_index(drop=True,inplace=True)
        
        # Population Average Time Series
        self.EvolAvg = self.EvolPop["Generation"].to_frame()
        avg = self.EvolPop.drop(columns="Generation").mean(axis=1,skipna=True)
        self.EvolAvg.insert(1,"Average",avg)

        return

#-------------------------------------------
#          Basic Plotting Functions
#-------------------------------------------
    def LoadAllPlotData(self,da=True,d2=True,d3=True):
        for i in self.Permutations:
            pm = int(i)
            if da == True:
                self.LoadSenderActivation(pm,1)
                self.LoadSenderActivation(pm,2)
                self.LoadReceiverActivation(pm,2)
                self.LoadReceiverActivation(pm,3)
                self.LoadSenderSensor(pm,1)
                self.LoadSenderSensor(pm,2)
                self.LoadReceiverSensor(pm,2)
                self.LoadReceiverSensor(pm,3)
            if d2 == True:
                self.LoadSenderPosition2D(pm,1)
                self.LoadSenderPosition2D(pm,2)
                self.LoadReceiverPosition2D(pm,2)
                self.LoadReceiverPosition2D(pm,3)
                self.LoadPosts2D(pm,1)
                self.LoadPosts2D(pm,3)
                self.LoadContact2D(pm,1)
                self.LoadContact2D(pm,2)
                self.LoadContact2D(pm,3)
            if d3 == True:
                self.LoadSenderPosition3D(pm,1)
                self.LoadSenderPosition3D(pm,2)
                self.LoadReceiverPosition3D(pm,2)
                self.LoadReceiverPosition3D(pm,3)
                self.LoadPosts3D(pm,1)
                self.LoadPosts3D(pm,3)
                self.LoadContact3D(pm,1)
                self.LoadContact3D(pm,2)
                self.LoadContact3D(pm,3)
        return
    
    def LoadSenderActivation(self,perm,phase):
        if phase > 2:
            raise KeyError("sender not available in phase 3")
        # simplify syntax
        pidx = self.PermutationIndex(perm)
        s = self.Sender[pidx][phase-1]
        # colors
        c_a = [[.251,.878,.816],[.351,.133,.545],[.312,.784,.471],[.941,.502,.502],[.769,.643,.518]]
        # add Line2D objects per neuron
        for i in range(5):
            self.ActLines[0][pidx][phase-1][i] = Line2D(
                self.Time[phase-1], s[self.NeuronIndex(i+1)],
                c=c_a[i], label=self.NeuronIndex(i+1)
                )
        return
    def PlotSenderActivation(self,ax,perm,phase,n=[1,2,3,4,5]):
        if phase > 2:
            raise KeyError("sender not available in phase 3")
        # simplify syntax
        pidx = self.PermutationIndex(perm)
        # load if necessary
        if any(self.ActLines[0][pidx][phase-1] == None):
            self.LoadSenderActivation(perm,phase)
        # add Line2D copies to Axes
        for i in n:    
            ax.add_line(copy(self.ActLines[0][pidx][phase-1][i-1]))
        ax.autoscale()
        return ax
    
    def LoadReceiverActivation(self,perm,phase):
        if phase < 2:
            raise KeyError("receiver not available in phase 1")
        # simplify syntax
        pidx = self.PermutationIndex(perm)
        r = self.Receiver[pidx][phase-2]
        # RGB color codes
        c_a = [[.251,.878,.816],[.351,.133,.545],[.312,.784,.471],[.941,.502,.502],[.769,.643,.518]]
        # Generate Line2D objects per neuron
        for i in range(5):
            self.ActLines[1][pidx][phase-2][i] = Line2D(
                self.Time[phase-1], r[self.NeuronIndex(i+1)],
                c=c_a[i], label=self.NeuronIndex(i+1)
                )
        return
    def PlotReceiverActivation(self,ax,perm,phase,n=[1,2,3,4,5]):
        if phase < 2:
            raise KeyError("receiver not available in phase 1")
        pidx = self.PermutationIndex(perm)
        # Load Lin2D objects of necessary
        if any(self.ActLines[1][pidx][phase-2] == None):
            self.LoadReceiverActivation(perm,phase)
        # add Line2D copies to Axes
        for i in n:
            ax.add_line(copy(self.ActLines[1][pidx][phase-2][i-1]))
        ax.autoscale()
        return ax
    
    def LoadSenderSensor(self,perm,phase):
        if phase > 2:
            raise KeyError("sender not available in phase 3")
        # simplify syntax
        pidx = self.PermutationIndex(perm)
        s = self.Sender[pidx][phase-1]["Sensor"]
        # Generate Line2D object for sensor
        self.SensLines[0][pidx][phase-1] = Line2D(
            self.Time[phase-1], s,
            c=[1,0,0], label="Sensor"
            )
        return
    def PlotSenderSensor(self,ax,perm,phase):
        if phase > 2:
            raise KeyError("sender not available in phase 3")
        pidx = self.PermutationIndex(perm)
        # Generate Line2D objects if necessary
        if self.SensLines[0][pidx][phase-1] == None:
            self.LoadSenderSensor(perm,phase)
        # Add Line2D copies to Axes
        ax.add_line(copy(self.SensLines[0][pidx][phase-1]))
        ax.autoscale()
        return ax
    
    def LoadReceiverSensor(self,perm,phase):
        if phase < 2:
            raise KeyError("receiver not available in phase 1")
        # simplify syntax
        pidx = self.PermutationIndex(perm)
        r = self.Receiver[pidx][phase-2]["Sensor"]
        # Generate Line2D object for sensor
        self.SensLines[1][pidx][phase-2] = Line2D(
            self.Time[phase-1], r,
            c=[1,0,0], label="Sensor"
            )
        return
    def PlotReceiverSensor(self,ax,perm,phase):
        if phase < 2:
            raise KeyError("receiver not available in phase 1")
        pidx = self.PermutationIndex(perm)
        # Generate Line2D object as necessary
        if self.SensLines[1][pidx][phase-2] == None:
            self.LoadReceiverSensor(perm,phase)
        # Add Line2D copies to Axes
        ax.add_line(copy(self.SensLines[1][pidx][phase-2]))
        ax.autoscale()
        return ax
    
    def LoadSenderPosition2D(self,perm,phase):
        if phase > 2:
            raise KeyError("sender not available in phase 3")
        # simplify syntax
        pidx = self.PermutationIndex(perm)
        s = self.Sender[pidx][phase-1]["Position"]
        # Generate Line2D object for position
        self.PosLines_2D[0][pidx][phase-1] = Line2D(
            self.Time[phase-1], s,
            c=[0,0,1], label="Sender"
            )
        return
    def PlotSenderPosition2D(self,ax,perm,phase):
        if phase > 2:
            raise KeyError("sender not available in phase 3")
        pidx = self.PermutationIndex(perm)
        # Generate Line2D objects as necesary
        if self.PosLines_2D[0][pidx][phase-1] == None:
            self.LoadSenderPosition2D(perm,phase)
        # Add Line2D copies to Axes 
        ax.add_line(copy(self.PosLines_2D[0][pidx][phase-1]))
        ax.autoscale()
        return ax
    
    def LoadReceiverPosition2D(self,perm,phase):
        if phase < 2:
            raise KeyError("receiver not available in phase 1")
        # simplify syntax
        pidx = self.PermutationIndex(perm)
        r = self.Receiver[pidx][phase-2]["Position"]
        # Generate Line2D objects for position
        self.PosLines_2D[1][pidx][phase-2] = Line2D(
            self.Time[phase-1], r,
            c=[1,.418,.706], label="Receiver"
            )
        return
    def PlotReceiverPosition2D(self,ax,perm,phase):
        if phase < 2:
            raise KeyError("receiver not available in phase 1")
        pidx = self.PermutationIndex(perm)
        # Generate Line2D objects if missing
        if self.PosLines_2D[1][pidx][phase-2] == None:
            self.LoadReceiverPosition2D(perm,phase)
        # Add Line2D copies to Axes
        ax.add_line(copy(self.PosLines_2D[1][pidx][phase-2]))
        ax.autoscale()
        return ax

    def LoadContact2D(self,perm,phase):
        # simplify syntax
        pidx = self.PermutationIndex(perm)
        t = self.Time[phase-1]
        # Select available agent
        if phase < 3: a = self.Sender[pidx][phase-1]
        else: a = self.Receiver[pidx][phase-2]
        # Lists to store contact points
        touch_t = []
        touch_pos = []
        # redundancy flag
        flag = 0
        # search through sensor time series
        for i in range(len(t)):
            # append contact point if sensor threshold met
            if (a["Sensor"][i] > 0.5 and flag == 0):
                touch_t.append(t[i])
                touch_pos.append(a["Position"][i])
                flag = 1
            # avoid adjacent successive contact points
            elif (a["Sensor"][i] > 0.5 and flag == 1):
                continue
            # reset flag to search for new contact points
            else: flag = 0
        # add contact points as Line2D object
        self.ContactPoints2D[pidx][phase-1] = Line2D(
            touch_t,touch_pos,
            c=[1,0,0],marker="o",markersize=3,linestyle="none",
            label="Contact"
            )
        return
    def PlotContact2D(self,ax,perm,phase):
        pidx = self.PermutationIndex(perm)
        # Generate Line2D objects as necessary
        if self.ContactPoints2D[pidx][phase-1] == None:
            self.LoadContact2D(perm,phase)
        # Add Line2D copies to Axes
        ax.add_line(copy(self.ContactPoints2D[pidx][phase-1]))
        ax.autoscale()
        return ax
    
    def LoadPosts2D(self,perm,phase):
        if phase == 2:
            raise KeyError("no posts in phase 2")
        # simplify syntax
        pix = int(phase==3)     # flag for alt. posts
        pidx = self.PermutationIndex(perm)
        t = self.Time[phase-1]
        dur = t[-1]
        pst = self.Posts[pidx]
        # list of line objects per post
        lines = []
        # track target posts
        assign = 1
        # iterate through posts
        for i in range(len(pst.columns)):
            # label target post
            if (i == 0):
                pos = np.full(int((dur/self.p["StepSize"][0])+1),pst[i][pix])
                lines.append(Line2D(t,pos,c=[1,.75,.275],linestyle="--",label="Target"))
                continue
            # add nearest posts as target
            elif (abs(pst[i-1][pix] - pst[i][pix]) < self.p["BodySize"][0]*3):
                pos = np.full(int((dur/self.p["StepSize"][0])+1),pst[i][pix])
                lines.append(Line2D(t,pos,c=[1,.75,.275],linestyle="--"))
                assign += 1
                continue
            break
        # load alt. posts for phase 3
        if (pix == 1):
            # iterate through remaining posts
            for i in range(assign, len(pst.columns)):
                pos = np.full(int((dur/self.p["StepSize"][0])+1),pst[i][pix])
                # label first Alt. post
                if (i == assign):
                    lines.append(Line2D(t,pos,c=[.035,.475,.412],linestyle="--",label="Alt."))
                    continue
                # add remaining posts as Alt.
                else:
                    lines.append(Line2D(t,pos,c=[.035,.475,.412],linestyle="--"))
                    continue
        # add list to object
        self.PostLines_2D[pidx][pix] = lines
        return
    def PlotPosts2D(self,ax,perm,phase):
        if phase == 2:
            raise KeyError("no posts in phase 2")
        pix = int(phase==3)
        pidx = self.PermutationIndex(perm)
        if (self.PostLines_2D[pidx][pix] == None):
            self.LoadPosts2D(perm,phase)
        lines = self.PostLines_2D[pidx][pix]
        for l in lines:
            ax.add_line(copy(l))
        ax.autoscale()
        return ax

    def LoadSenderPosition3D(self,perm,phase):
        if phase > 2:
            raise KeyError("sender not available in phase 3")
        # simplify syntax
        pidx = self.PermutationIndex(perm)
        # decompose position time series into 3 coordinates
        s = XSinCos(self.Time[phase-1],self.Sender[pidx][phase-1]["Position"])
        # Generate Line3D object
        self.PosLines_3D[0][pidx][phase-1] = Line3D(
            s[0], s[1], s[2],
            c=[0,0,1], label="Sender", linestyle="-",linewidth=2
            )
        return
    def PlotSenderPosition3D(self,ax,perm,phase):
        if phase > 2:
            raise KeyError("sender not available in phase 3")
        # simplify syntax
        t = self.Time[phase-1]
        dur = t[-1]
        ax.set_xlim([0,dur])
        pidx = self.PermutationIndex(perm)
        # Generate Line3D plots if necessary
        if self.PosLines_3D[0][pidx][phase-1] == None:
            self.LoadSenderPosition3D(perm,phase)
        # Add Line2D copies to Axes
        ax.add_line(copy(self.PosLines_3D[0][pidx][phase-1]))
        return ax
    
    def LoadReceiverPosition3D(self,perm,phase):
        if phase < 2:
            raise KeyError("receiver not available in phase 1")
        # simplify syntax
        pidx = self.PermutationIndex(perm)
        # decompose position time series into 3 coordinates
        r = XSinCos(self.Time[phase-1],self.Receiver[pidx][phase-2]["Position"])
        # Generate Line3D object
        self.PosLines_3D[1][pidx][phase-2] = Line3D(
            r[0], r[1], r[2],
            c=[1,.418,.706], label="Receiver", linestyle="-",linewidth=2
            )
        return
    def PlotReceiverPosition3D(self,ax,perm,phase):
        if phase < 2:
            raise KeyError("receiver not available in phase 1")
        # simplify syntax
        t = self.Time[phase-1]
        dur = t[-1]
        ax.set_xlim([0,dur])
        pidx = self.PermutationIndex(perm)
        # Generate Line3D plots if necessary
        if self.PosLines_3D[1][pidx][phase-2] == None:
            self.LoadReceiverPosition3D(perm,phase)
        # Add Line2D copies to Axes
        ax.add_line(copy(self.PosLines_3D[1][pidx][phase-2]))
        return ax
    
    def LoadPosts3D(self,perm,phase):
        if phase == 2:
            raise KeyError("no posts in phase 2")
        # simplify syntax
        pix = int(phase==3) # flag for Alt. posts
        pidx = self.PermutationIndex(perm)
        t = self.Time[phase-1]
        pst = self.Posts[pidx]
        # list of line objects per post
        lines = []
        # track target posts
        assign = 1
        # iterate through posts
        for i in range(len(pst.columns)):
            # label first target post
            if (i == 0):
                pos = XSinCos(t,pst[i][pix])
                lines.append(Line3D(pos[0],pos[1],pos[2],c=[1,.75,.275],linestyle="--",linewidth=2,label="Target"))
                continue
            # add nearest posts as target
            elif (abs(pst[i-1][pix] - pst[i][pix]) < self.p["BodySize"][0]*3):
                pos = XSinCos(t,pst[i][pix])
                lines.append(Line3D(pos[0],pos[1],pos[2],c=[1,.75,.275],linestyle="--",linewidth=2))
                assign += 1
                continue
            break
        # load alt. posts for phase 3
        if (pix == 1):
            # iterate through remaining posts
            for i in range(assign, len(pst.columns)):
                pos = XSinCos(t,pst[i][pix])
                # label first Alt. post
                if (i == assign):
                    lines.append(Line3D(pos[0],pos[1],pos[2],c=[.035,.475,.412],linestyle="--",linewidth=2,label="Alt."))
                    continue
                # add remaining posts as Alt.
                else:
                    lines.append(Line3D(pos[0],pos[1],pos[2],c=[.035,.475,.412],linestyle="--",linewidth=2))
                    continue
        # add Line3D list to object
        self.PostLines_3D[pidx][pix] = lines
        return
    def PlotPosts3D(self,ax,perm,phase):
        if phase == 2:
            raise KeyError("no posts in phase 2")
        # simplify syntax
        t = self.Time[phase-1]
        dur = t[-1]
        ax.set_xlim([0,dur])
        pix = int(phase==3)
        pidx = self.PermutationIndex(perm)
        # Generate Line3D object if necessary
        if (self.PostLines_3D[pidx][pix] == None):
            self.LoadPosts3D(perm,phase)
        # add Line3D copies to Axes
        lines = self.PostLines_3D[pidx][pix]
        for l in lines:
            ax.add_line(copy(l))
        return ax
      
    def LoadContact3D(self,perm,phase):
        # simplify snytax
        pidx = self.PermutationIndex(perm)
        t = self.Time[phase-1]
        # select available target
        if phase < 3: a = self.Sender[pidx][phase-1]
        else: a = self.Receiver[pidx][phase-2]
        # store contact points per coordinate
        touch_x = []
        touch_y = []
        touch_z = []
        # redundancy flag
        flag = 0
        # search through sensor time series
        for i in range(len(t)):
            pos = a["Position"][i]
            # append contact point if sensor threshold met
            if (a["Sensor"][i] > 0.5 and flag == 0):
                touch_x.append(t[i])
                touch_y.append(np.cos(pos/100))
                touch_z.append(np.sin(pos/100))
                flag = 1
            # avoid adjacent successive contact points
            elif (a["Sensor"][i] > 0.5 and flag == 1):
                continue
            # reset flag to search for new contact points
            elif flag == 1: flag = 0
        # add contact points as Line2D object
        self.ContactPoints3D[pidx][phase-1] = Line3D(
            touch_x,touch_y,touch_z,
            c=[1,0,0],marker="o",markersize=6,linestyle="none",
            label="Contact"
        )
        return
    def PlotContact3D(self,ax,perm,phase):
        # simplify snytax
        pidx = self.PermutationIndex(perm)
        # Generate Line3D objects if necessary
        if self.ContactPoints3D[pidx][phase-1] == None:
            self.LoadContact3D(perm,phase)
        # add Line3D copies to Axes
        ax.add_line(copy(self.ContactPoints3D[pidx][phase-1]))
        return ax
     
    def PlotScaffold(self,ax,n=2,lyz=False,lx=True):
        # circle on y-z plane
        cy = np.cos(np.linspace(0,2*np.pi,num=100))
        cz = np.sin(np.linspace(0,2*np.pi,num=100))
        # fill plot with circles
        for i in range(n):
            cx = i * (ax.get_xlim()[1] / (n-1))
            ax.plot(cy,cz,zs=cx,zdir="x",c="grey",linestyle=":")
        # crossing lines at first and last circle
        if lyz == True:
            ax.plot(np.full(100,0),np.linspace(-1,1,num=100),zs=0,zdir="x",c="grey",linestyle=":")
            ax.plot(np.linspace(-1,1,num=100),np.full(100,0),zs=0,zdir="x",c="grey",linestyle=":")
            ax.plot(np.full(100,0),np.linspace(-1,1,num=100),zs=ax.get_xlim()[1],zdir="x",c="grey",linestyle=":")
            ax.plot(np.linspace(-1,1,num=100),np.full(100,0),zs=ax.get_xlim()[1],zdir="x",c="grey",linestyle=":")
        # 4 lines through the circles
        if lx == True:
            ax.plot(np.linspace(0,ax.get_xlim()[1],num=100),
                    np.full(100,np.cos(np.pi/4)),np.full(100,np.sin(np.pi/4)),
                    c="grey",linestyle=":")
            ax.plot(np.linspace(0,ax.get_xlim()[1],num=100),
                    -np.full(100,np.cos(np.pi/4)),np.full(100,np.sin(np.pi/4)),
                    c="grey",linestyle=":")
            ax.plot(np.linspace(0,ax.get_xlim()[1],num=100),
                    -np.full(100,np.cos(np.pi/4)),-np.full(100,np.sin(np.pi/4)),
                    c="grey",linestyle=":")
            ax.plot(np.linspace(0,ax.get_xlim()[1],num=100),
                    np.full(100,np.cos(np.pi/4)),-np.full(100,np.sin(np.pi/4)),
                    c="grey",linestyle=":")
        return ax

#-------------------------------------------
#          Preset Plotting Functions
#-------------------------------------------
    def Plot3D2D_Pos (self,perm,size=(20,10)):
        fig = plt.figure(figsize=size)
        
        # 3D Phase 1
        ax = fig.add_subplot(2,3,1,projection="3d")
        ax.set_xlim([0,self.Time[0][-1]])
        ax.set_xticks([0,self.Time[0][-1]])
        ax.tick_params(labelsize=16)
        ax = ClearGrid(ax)
        ax = self.PlotSenderPosition3D(ax,perm=perm,phase=1)
        ax = self.PlotPosts3D(ax,perm=perm,phase=1)
        ax = self.PlotScaffold(ax,n=30)
        ax = self.PlotContact3D(ax,perm=perm,phase=1)
        h1, l1 = ax.get_legend_handles_labels()
        ax.set_box_aspect(aspect=[1,.4,.4])

        # 3D Phase 2
        ax = fig.add_subplot(2,3,2,projection="3d")
        ax.set_xlim([0,self.Time[1][-1]])
        ax.set_xticks([0,self.Time[1][-1]])
        ax.tick_params(labelsize=16)
        ax = ClearGrid(ax)
        ax = self.PlotSenderPosition3D(ax,perm=perm,phase=2)
        ax = self.PlotReceiverPosition3D(ax,perm=perm,phase=2)
        ax = self.PlotScaffold(ax,n=30)
        ax = self.PlotContact3D(ax,perm=perm,phase=2)
        h2, l2 = ax.get_legend_handles_labels()
        ax.set_box_aspect(aspect=[1,.4,.4])

        # 3D Phase 1
        ax = fig.add_subplot(2,3,3,projection="3d")
        ax.set_xlim([0,self.Time[2][-1]])
        ax.set_xticks([0,self.Time[2][-1]])
        ax.tick_params(labelsize=16)
        ax = ClearGrid(ax)
        ax = self.PlotReceiverPosition3D(ax,perm=perm,phase=3)
        ax = self.PlotPosts3D(ax,perm=perm,phase=3)
        ax = self.PlotScaffold(ax,n=30)
        ax = self.PlotContact3D(ax,perm=perm,phase=3)
        h3, l3 = ax.get_legend_handles_labels()
        ax.set_box_aspect(aspect=[1,.4,.4])

        # 2D Phase 1
        ax = fig.add_subplot(2,3,4)
        ax.set_xlim([0,self.Time[0][-1]])
        ax.set_ylim([0,2*np.pi])
        ax.set_xticks([0,self.Time[0][-1]])
        ax.set_yticks([0,614])
        ax.tick_params(labelsize=16)
        ax.set_xlabel("Time",fontsize=16)
        ax.set_ylabel("Position",fontsize=16)
        ax = self.PlotSenderPosition2D(ax,perm=perm,phase=1)
        ax = self.PlotPosts2D(ax,perm=perm,phase=1)
        ax = self.PlotContact2D(ax,perm=perm,phase=1)
        ax.set_box_aspect(aspect=.5)

        # 2D Phase 2
        ax = fig.add_subplot(2,3,5)
        ax.set_xlim([0,self.Time[1][-1]])
        ax.set_ylim([0,2*np.pi])
        ax.set_xticks([0,self.Time[1][-1]])
        ax.set_yticks([])
        ax.tick_params(labelsize=16)
        ax.set_xlabel("Time",fontsize=16)
        ax = self.PlotSenderPosition2D(ax,perm=perm,phase=2)
        ax = self.PlotReceiverPosition2D(ax,perm=perm,phase=2)
        ax = self.PlotContact2D(ax,perm=perm,phase=2)
        ax.set_box_aspect(aspect=.5)

        # 2D Phase 1
        ax = fig.add_subplot(2,3,6)
        ax.set_xlim([0,self.Time[2][-1]])
        ax.set_ylim([0,2*np.pi])
        ax.set_xticks([0,self.Time[2][-1]])
        ax.set_yticks([])
        ax.tick_params(labelsize=16)
        ax.set_xlabel("Time",fontsize=16)
        ax = self.PlotReceiverPosition2D(ax,perm=perm,phase=3)
        ax = self.PlotPosts2D(ax,perm=perm,phase=3)
        ax = self.PlotContact2D(ax,perm=perm,phase=3)
        ax.set_box_aspect(aspect=.5)

        # prevent redundancy in legend
        h = h1 + h2 + h3
        l = l1 + l2 + l3
        by_label = dict(zip(l,h))

        return fig, ax, by_label
    
    def Plot3D2D_PosAct (self,perm,size=(20,15)):
        fig = plt.figure(figsize=size)
        gs = gridspec.GridSpec(ncols=3,nrows=3, figure=fig,height_ratios=[2,1,1])
        aspect2d = .5 # aspect ratio for 2D activation plots
        
        # 3D Phase 1
        ax = fig.add_subplot(gs[0,0],projection="3d")
        ax.set_xlim([0,self.Time[0][-1]])
        ax.set_xticks([0,self.Time[0][-1]])
        ax.tick_params(labelsize=16)
        ax = ClearGrid(ax)
        ax = self.PlotSenderPosition3D(ax,perm=perm,phase=1)
        ax = self.PlotPosts3D(ax,perm=perm,phase=1)
        ax = self.PlotScaffold(ax,n=30)
        ax = self.PlotContact3D(ax,perm=perm,phase=1)
        h1, l1 = ax.get_legend_handles_labels()
        ax.set_box_aspect(aspect=[1,.4,.4])

        # 3D Phase 2
        ax = fig.add_subplot(gs[0,1],projection="3d")
        ax.set_xlim([0,self.Time[1][-1]])
        ax.set_xticks([0,self.Time[1][-1]])
        ax.tick_params(labelsize=16)
        ax = ClearGrid(ax)
        ax = self.PlotSenderPosition3D(ax,perm=perm,phase=2)
        ax = self.PlotReceiverPosition3D(ax,perm=perm,phase=2)
        ax = self.PlotScaffold(ax,n=30)
        ax = self.PlotContact3D(ax,perm=perm,phase=2)
        h2, l2 = ax.get_legend_handles_labels()
        ax.set_box_aspect(aspect=[1,.4,.4])

        # 3D Phase 1
        ax = fig.add_subplot(gs[0,2],projection="3d")
        ax.set_xlim([0,self.Time[2][-1]])
        ax.set_xticks([0,self.Time[2][-1]])
        ax.tick_params(labelsize=16)
        ax = ClearGrid(ax)
        ax = self.PlotReceiverPosition3D(ax,perm=perm,phase=3)
        ax = self.PlotPosts3D(ax,perm=perm,phase=3)
        ax = self.PlotScaffold(ax,n=30)
        ax = self.PlotContact3D(ax,perm=perm,phase=3)
        h3, l3 = ax.get_legend_handles_labels()
        ax.set_box_aspect(aspect=[1,.4,.4])

        # 2D Phase 1 -- Sender
        ax = fig.add_subplot(gs[1,0])
        ax.set_xlim([0,self.Time[0][-1]])
        ax.set_ylim([-.1,1.1])
        ax.set_xticks([])
        ax.set_yticks([0,1])
        ax.tick_params(labelsize=16)
        ax.set_ylabel("Activation",fontsize=16)
        ax = self.PlotSenderActivation(ax,perm=perm,phase=1)
        h4, l4 = ax.get_legend_handles_labels()
        ax.set_box_aspect(aspect=aspect2d)

        # 2D Phase 1 -- Empty
        ax = fig.add_subplot(gs[2,0])
        ax.set_xlim([0,self.Time[0][-1]])
        ax.set_ylim([-.1,1.1])
        ax.set_xticks([0,self.Time[0][-1]])
        ax.set_yticks([0,1])
        ax.tick_params(labelsize=16)
        ax.set_xlabel("Time",fontsize=16)
        ax.set_ylabel("Activation",fontsize=16)
        ax.plot(np.linspace(0,250,num=3),np.full(3,0),c="white")
        ax.plot(np.linspace(0,250,num=3),np.full(3,1),c="white")
        ax.autoscale()
        ax.set_box_aspect(aspect=aspect2d)

        # 2D Phase -- Sender 2
        ax = fig.add_subplot(gs[1,1])
        ax.set_xlim([0,self.Time[1][-1]])
        ax.set_ylim([-.1,1.1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax = self.PlotSenderActivation(ax,perm=perm,phase=2)
        ax.set_box_aspect(aspect=aspect2d)

        # 2D Phase -- Receiver 2
        ax = fig.add_subplot(gs[2,1])
        ax.set_xlim([0,self.Time[1][-1]])
        ax.set_ylim([-.1,1.1])
        ax.set_xticks([0,self.Time[1][-1]])
        ax.set_yticks([])
        ax.tick_params(labelsize=16)
        ax.set_xlabel("Time",fontsize=16)
        ax = self.PlotReceiverActivation(ax,perm=perm,phase=2)
        ax.set_box_aspect(aspect=aspect2d)

        # 2D Phase -- Empty 2
        ax = fig.add_subplot(gs[1,2])
        ax.set_xlim([0,self.Time[2][-1]])
        ax.set_ylim([-.1,1.1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(aspect=aspect2d)

        # 2D Phase 3 -- Receiver
        ax = fig.add_subplot(gs[2,2])
        ax.set_xlim([0,self.Time[2][-1]])
        ax.set_ylim([-.1,1.1])
        ax.set_xticks([0,self.Time[2][-1]])
        ax.set_yticks([])
        ax.tick_params(labelsize=16)
        ax.set_xlabel("Time",fontsize=16)
        ax = self.PlotReceiverActivation(ax,perm=perm,phase=3)
        ax.set_box_aspect(aspect=aspect2d)

        # prevent redundancy in legend
        h = h1 + h2 + h3 + h4
        l = l1 + l2 + l3 + l4
        by_label = dict(zip(l,h))

        return fig, ax, by_label
    
    def Plot3D2D_PosSen (self,perm,size=(20,10)):
        fig = plt.figure(figsize=size)
        gs=gridspec.GridSpec(ncols=3,nrows=2,figure=fig,height_ratios=[3,1])
        aspect2d=.4 # aspect ratio for 2D sensor plots
        
        # 3D Phase 1
        ax = fig.add_subplot(gs[0,0],projection="3d")
        ax.set_xlim([0,self.Time[0][-1]])
        ax.set_xticks([0,self.Time[0][-1]])
        ax.tick_params(labelsize=16)
        ax = ClearGrid(ax)
        ax = self.PlotSenderPosition3D(ax,perm=perm,phase=1)
        ax = self.PlotPosts3D(ax,perm=perm,phase=1)
        ax = self.PlotScaffold(ax,n=30)
        ax = self.PlotContact3D(ax,perm=perm,phase=1)
        h1, l1 = ax.get_legend_handles_labels()
        ax.set_box_aspect(aspect=[1,.4,.4])

        # 3D Phase 2
        ax = fig.add_subplot(gs[0,1],projection="3d")
        ax.set_xlim([0,self.Time[1][-1]])
        ax.set_xticks([0,self.Time[1][-1]])
        ax.tick_params(labelsize=16)
        ax = ClearGrid(ax)
        ax = self.PlotSenderPosition3D(ax,perm=perm,phase=2)
        ax = self.PlotReceiverPosition3D(ax,perm=perm,phase=2)
        ax = self.PlotScaffold(ax,n=30)
        ax = self.PlotContact3D(ax,perm=perm,phase=2)
        ax.set_box_aspect(aspect=[1,.4,.4])

        # 3D Phase 1
        ax = fig.add_subplot(gs[0,2],projection="3d")
        ax.set_xlim([0,self.Time[2][-1]])
        ax.set_xticks([0,self.Time[2][-1]])
        ax.tick_params(labelsize=16)
        ax = ClearGrid(ax)
        ax = self.PlotReceiverPosition3D(ax,perm=perm,phase=3)
        ax = self.PlotPosts3D(ax,perm=perm,phase=3)
        ax = self.PlotScaffold(ax,n=30)
        ax = self.PlotContact3D(ax,perm=perm,phase=3)
        h2, l2 = ax.get_legend_handles_labels()
        ax.set_box_aspect(aspect=[1,.4,.4])

        # 2D Phase 1
        ax = fig.add_subplot(gs[1,0])
        ax.set_xlim([0,self.Time[0][-1]])
        ax.set_xticks([0,self.Time[0][-1]])
        ax.set_yticks([])
        ax.tick_params(labelsize=16)
        ax.set_xlabel("Time",fontsize=16)
        ax.set_ylabel("Sensor",fontsize=16)
        ax = self.PlotSenderSensor(ax,perm=perm,phase=1)
        ax.set_box_aspect(aspect=aspect2d)
        h3, l3 = ax.get_legend_handles_labels()

        # 2D Phase 2
        ax = fig.add_subplot(gs[1,1])
        ax.set_xlim([0,self.Time[1][-1]])
        ax.set_xticks([0,self.Time[1][-1]])
        ax.set_yticks([])
        ax.tick_params(labelsize=16)
        ax.set_xlabel("Time",fontsize=16)
        ax = self.PlotSenderSensor(ax,perm=perm,phase=2)
        ax.set_box_aspect(aspect=aspect2d)

        # 2D Phase 1
        ax = fig.add_subplot(gs[1,2])
        ax.set_xlim([0,self.Time[2][-1]])
        ax.set_xticks([0,self.Time[2][-1]])
        ax.set_yticks([])
        ax.tick_params(labelsize=16)
        ax.set_xlabel("Time",fontsize=16)
        ax = self.PlotReceiverSensor(ax,perm=perm,phase=3)
        ax.set_box_aspect(aspect=aspect2d)

        # prevent redundancy in legend
        h = h1 + h2 + h3
        l = l1 + l2 + l3
        by_label = dict(zip(l,h))

        return fig, ax, by_label

    def PlotEvo (self,ax, gb=[0,10000],mark=False,leg=True):
        
        best, = ax.plot(self.EvolBest["Generation"][gb[0]:gb[1]], self.EvolBest["Best"][gb[0]:gb[1]],
                        c=[0,0,1],label="Best",linewidth=.5)
        avg, = ax.plot(self.EvolAvg["Generation"][gb[0]:gb[1]], self.EvolAvg["Average"][gb[0]:gb[1]],
                       c=[1,.75,.275],label="Average",linewidth=.5)
        
        handles = [best,avg]
        
        # mark final fitness with horizontal red line
        if mark == True:
            val = self.TotalFitness
            final, = ax.plot(np.linspace(gb[0],gb[1],num=50),np.full(50, val),
                             c=[1,0,0],linestyle="--",
                             label=("Final: " + str(val)))

        if leg == True:
            ax.legend(handles=handles,loc="center right",fontsize=16)
        
        return ax

#---------------------------
#       Plot Helpers
#---------------------------
# cleans 3D plots
def ClearGrid(ax):
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.pane.set_facecolor(".75")
    ax.yaxis.pane.set_facecolor(".75")
    ax.zaxis.pane.set_facecolor(".75")
    return ax

# converts 2D points/arrays to 3D coordinates
def XSinCos(x,pos):
    if type(pos) == np.float64:
        cos = np.cos(np.full(len(x),pos) / 100)
        sin = np.sin(np.full(len(x),pos) / 100)
    else:
        cos = np.cos(pos / 100)
        sin = np.sin(pos / 100)
    return [x, cos, sin]