#!/usr/bin/env wolframscript
(* ::Package:: *)

(* ::Title::Italic:: *)
(*Analysis of a Referential Communication Task*)


(* ::Subtitle::Italic:: *)
(*Thomas Gaul*)


(* ::Section::Initialization::Closed:: *)
(*(*Initialisation*)*)


(* If you haven't already, download Dynamica from https://rdbeer.pages.iu.edu 
and put it in a folder that Mathematica can find *)


(* ::Input::Initialization:: *)
Get["Dynamica`"];


(* ::Subsection::Initialization:: *)
(*(*Definitions for Loading Phenotype*)*)


(* ::Input::Initialization:: *)
$ParentDirectory = FileNameJoin[{NotebookDirectory[],"..","EvoP2"}];
SetDirectory[$ParentDirectory];
fileName[pop_]:=FileNameJoin[{StringJoin["Pop-",ToString[pop]],"phenotype.dat"}];
SetPhenotype[pop_]:=Module[{data,taus,thetas,weights,sensorweights},data=Flatten[Import[fileName[pop],"Table"]];
taus=data[[1;;5]];
thetas=data[[6;;10]];
weights=Partition[data[[11;;35]],5];
sensorweights=data[[36;;]];
CTRNN5y/.{
\[Tau]1->taus[[1]],\[Tau]2->taus[[2]],\[Tau]3->taus[[3]],\[Tau]4->taus[[4]],\[Tau]5->taus[[5]],
\[Theta]1->thetas[[1]],\[Theta]2->thetas[[2]],\[Theta]3->thetas[[3]],\[Theta]4->thetas[[4]],\[Theta]5->thetas[[5]],
w11->weights[[1,1]],w21->weights[[2,1]],w31->weights[[3,1]],w41->weights[[4,1]],w51->weights[[5,1]],
w12->weights[[1,2]],w22->weights[[2,2]],w32->weights[[3,2]],w42->weights[[4,2]],w52->weights[[5,2]],
w13->weights[[1,3]],w23->weights[[2,3]],w33->weights[[3,3]],w43->weights[[4,3]],w53->weights[[5,3]],
w14->weights[[1,4]],w24->weights[[2,4]],w34->weights[[3,4]],w44->weights[[4,4]],w54->weights[[5,4]],
w15->weights[[1,5]],w25->weights[[2,5]],w35->weights[[3,5]],w45->weights[[4,5]],w55->weights[[5,5]],
sw1->sensorweights[[1]],sw2->sensorweights[[2]],sw3->sensorweights[[3]],sw4->sensorweights[[4]],sw5->sensorweights[[5]]
}
];


(* ::Subsection::Initialization:: *)
(*(*CTRNN Definitions*)*)


(* ::Input::Initialization:: *)
\[Sigma][x_]:=1/(1+E^-x);


(* ::Subsubsection:: *)
(*5-Dimensional System*)


(* if the following cell takes to long, edit Dynamica.m and lower the
$DynamicaSymbolicThreshold (the default is 5). Use this line to verify. *)
$DynamicaSymbolicThreshold


(* ::Input::Initialization:: *)
CTRNN5y=DynamicalSystem[
{
(-y1+w11 \[Sigma][y1+\[Theta]1]+w21 \[Sigma][y2+\[Theta]2]+w31 \[Sigma][y3+\[Theta]3]+w41 \[Sigma][y4+\[Theta]4]+w51 \[Sigma][y5+\[Theta]5]+sw1 d)/\[Tau]1,
(-y2+w12 \[Sigma][y1+\[Theta]1]+w22 \[Sigma][y2+\[Theta]2]+w32 \[Sigma][y3+\[Theta]3]+w42 \[Sigma][y4+\[Theta]4]+w52 \[Sigma][y5+\[Theta]5]+sw2 d)/\[Tau]2,
(-y3+w13 \[Sigma][y1+\[Theta]1]+w23 \[Sigma][y2+\[Theta]2]+w33 \[Sigma][y3+\[Theta]3]+w43 \[Sigma][y4+\[Theta]4]+w53 \[Sigma][y5+\[Theta]5]+sw3 d)/\[Tau]3,
(-y4+w14 \[Sigma][y1+\[Theta]1]+w24 \[Sigma][y2+\[Theta]2]+w34 \[Sigma][y3+\[Theta]3]+w44 \[Sigma][y4+\[Theta]4]+w54 \[Sigma][y5+\[Theta]5]+sw4 d)/\[Tau]4,
(-y5+w15 \[Sigma][y1+\[Theta]1]+w25 \[Sigma][y2+\[Theta]2]+w35 \[Sigma][y3+\[Theta]3]+w45 \[Sigma][y4+\[Theta]4]+w55 \[Sigma][y5+\[Theta]5]+sw5 d)/\[Tau]5
},
{{y1,-35,35},{y2,-35,35},{y3,-35,35},{y4,-35,35},{y5,-35,35}},
{
{\[Theta]1,-16,16},{\[Theta]2,-16,16},{\[Theta]3,-16,16},{\[Theta]4,-16,16},{\[Theta]5,-16,16},
{\[Tau]1,1,30},{\[Tau]2,1,30},{\[Tau]3,1,30},{\[Tau]4,1,30},{\[Tau]5,1,30},
{sw1,-16,16},{sw2,-16,16},{sw3,-16,16},{sw4,-16,16},{sw5,-16,16},
{w11,-16,16},{w21,-16,16},{w31,-16,16},{w41,-16,16},{w51,-16,16},
{w12,-16,16},{w22,-16,16},{w32,-16,16},{w42,-16,16},{w52,-16,16},
{w13,-16,16},{w23,-16,16},{w33,-16,16},{w43,-16,16},{w53,-16,16},
{w14,-16,16},{w24,-16,16},{w34,-16,16},{w44,-16,16},{w54,-16,16},
{w15,-16,16},{w25,-16,16},{w35,-16,16},{w45,-16,16},{w55,-16,16},
{d,0,1}
}
];


(* ::Subsubsection:: *)
(*2-Dimensional Reduction in State Space*)


(* ::Input::Initialization:: *)
CTRNN2y=DynamicalSystem[
{(-y1+w11 \[Sigma][y1+\[Theta]1]+w21 \[Sigma][y2+\[Theta]2]+w41 +sw1 d)/\[Tau]1,
(-y2+w12 \[Sigma][y1+\[Theta]1]+w22 \[Sigma][y2+\[Theta]2]+w42 +sw2 d)/\[Tau]2},
{{y1,-35,35},{y2,-35,35}},
{
{w11,-16,16},{w21,-16,16},{w12,-16,16},{w22,-16,16},
{w41,-16,16},{w42,-16,16},
{sw1,-16,16},{sw2,-16,16},
{\[Tau]1,1,30},{\[Tau]2,1,30},
{\[Theta]1,-16,16},{\[Theta]2,-16,16},
{d,0,1}
}
];


(* ::Subsection:: *)
(*Loading Agent Phenotype*)


(* ::Input::Initialization:: *)
(* using agent 1 for most analysis *)
A = SetPhenotype[1];
A2d=CTRNN2y/.((#->DSParameterValue[A,#])&/@{
w11,w21,w12,w22,w41,w42,
sw1,sw2,
\[Tau]1,\[Tau]2,
\[Theta]1,\[Theta]2
}
);


(* ::Subsection:: *)
(*Transformation of Reduced Form to Output Space*)


(* ::Input::Initialization:: *)
With[{\[CurlyEpsilon]=10^-5},
Ao2d=TransformDynamicalSystem[A2d,{{o1,\[CurlyEpsilon],1-\[CurlyEpsilon]},{o2,\[CurlyEpsilon],1-\[CurlyEpsilon]}},{\[Sigma][y1+\[Theta]1],\[Sigma][y2+\[Theta]2]}]
];


(* ::Section::Closed:: *)
(*Phase Plane Analysis*)


(* ::Subsection:: *)
(*Time Scale in 5D System*)


(* ::Input:: *)
(*(* Time-constants *)*)
(*Print[#]&/@Table[A[[9,i]],{i,-5,-1}];*)


(* ::Input:: *)
(*A2d[[9]]*)


(* ::Subsubsection:: *)
(*Timescale near saddle manifold*)


(* ::Input:: *)
(*flow=Flatten[Table[{12+i,-4.9-j},{i,-1,.7,.1},{j,-.2,2,2.1}],1];*)
(*window={{11.4,14},{-6.5,-4.75}};*)
(*SaddlePoint=EquilibriumPoints[A/.{d->0}][[2,3]];*)
(*Show[*)
(*DisplayTrajectories[A2d/.{d->0},flow,145,StateVariables->{y1,y2},PlotRange->window,Arrows->1,PlotStyle->Black,FrameLabel->{"\!\(\*TemplateBox[<|\"boxes\" -> FormBox[SubscriptBox[StyleBox[\"N\", \"TI\"], \"1\"], TraditionalForm], \"errors\" -> {}, \"input\" -> \"N_1\", \"state\" -> \"Boxes\"|>,\n\"TeXAssistantTemplate\"]\)","\!\(\*TemplateBox[<|\"boxes\" -> FormBox[SubscriptBox[StyleBox[\"N\", \"TI\"], \"2\"], TraditionalForm], \"errors\" -> {}, \"input\" -> \"N_2\", \"state\" -> \"Boxes\"|>,\n\"TeXAssistantTemplate\"]\)"},LabelStyle->{FontSize->14},FrameTicksStyle->Directive[FontFamily->"Latin Modern Math",FontSize->12]*)
(*],*)
(*DisplayTrajectories[A2d/.{d->0},{{12.8,-4.6},{12.8,-6.9}},145,StateVariables->{y1,y2},PlotRange->window,Arrows->1,PlotStyle->LightGray],*)
(*DisplayPhasePortrait[A2d/.{d->0},StateVariables->{y1,y2},Tries->100,PlotRange->window],*)
(*ImageSize->Medium*)
(*]*)
(**)


(* ::Input:: *)
(*bounds={{11.99,13.25},{-6,-4.85}};*)
(*{dy1L,dy1U}=#[{Abs[A2d[[2,1]]/.Join[A2d[[9]],{d->0}]],bounds[[1,1]]<y1<bounds[[1,2]],bounds[[2,1]]<y2<bounds[[2,2]]},{y1,y2}]&/@{MinValue,MaxValue};*)
(*Show[*)
(*VectorPlot[A2d[[2]]/.Join[A2d[[9]],{d->0}],{y1,Sequence@@bounds[[1]]},{y2,Sequence@@bounds[[2]]}*)
(*,VectorPoints->Fine*)
(*,VectorColorFunction->Function[{x,y,vx,vy,n},ColorData["BrownCyanTones"][vx]]*)
(*,VectorColorFunctionScaling->{False,False,True,False,False}*)
(*,PlotLegends->Placed[BarLegend[*)
(*{ColorData[{"BrownCyanTones",{dy1L,dy1U}}][#]&,{dy1L,dy1U}},*)
(*LegendLabel->"\!\(\*TemplateBox[<|\"boxes\" -> FormBox[OverscriptBox[SubscriptBox[StyleBox[\"N\", \"TI\"], \"1\"], \".\"], TraditionalForm], \"errors\" -> {}, \"input\" -> \"\\\\dot{N_1}\", \"state\" -> \"Boxes\"|>,\n\"TeXAssistantTemplate\"]\)",*)
(*LegendMarkerSize->275,*)
(*LegendMargins->0,*)
(*LabelStyle->Directive[FontFamily->"Latin Modern Math",FontSize->12]*)
(*],Right]*)
(*,FrameLabel->{"\!\(\*TemplateBox[<|\"boxes\" -> FormBox[SubscriptBox[StyleBox[\"N\", \"TI\"], \"1\"], TraditionalForm], \"errors\" -> {}, \"input\" -> \"N_1\", \"state\" -> \"Boxes\"|>,\n\"TeXAssistantTemplate\"]\)","\!\(\*TemplateBox[<|\"boxes\" -> FormBox[SubscriptBox[StyleBox[\"N\", \"TI\"], \"2\"], TraditionalForm], \"errors\" -> {}, \"input\" -> \"N_2\", \"state\" -> \"Boxes\"|>,\n\"TeXAssistantTemplate\"]\)"}*)
(*,LabelStyle->{FontSize->14}*)
(*,FrameTicksStyle->Directive[FontFamily->"Latin Modern Math",FontSize->12]*)
(*]*)
(*,Quiet[DisplayPhasePortrait[A2d/.{d->0},Tries->300*)
(*,SaddleManifolds->True*)
(*,SaddleManifoldPlotPoints->1000*)
(*,SaddleManifoldDuration->5000*)
(*]]*)
(*]*)
(**)


(* ::Input:: *)
(**)


(* ::Subsection:: *)
(*Reduced Form in State Space*)


(* ::Input:: *)
(*(* ignore error messages from `NDSolve` *)*)
(*Quiet[DisplayPhasePortrait[A2d/.{d->0},Tries->300,PlotRange->{{5,30},{-25,0}},*)
(*NullManifolds->True,*)
(*SaddleManifolds->True,*)
(*SaddleManifoldPlotPoints->1000,*)
(*SaddleManifoldDuration->5000,*)
(*FrameLabel->{"\!\(\*TemplateBox[<|\"boxes\" -> FormBox[SubscriptBox[StyleBox[\"N\", \"TI\"], \"1\"], TraditionalForm], \"errors\" -> {}, \"input\" -> \"N_1\", \"state\" -> \"Boxes\"|>,\n\"TeXAssistantTemplate\"]\)","\!\(\*TemplateBox[<|\"boxes\" -> FormBox[SubscriptBox[StyleBox[\"N\", \"TI\"], \"2\"], TraditionalForm], \"errors\" -> {}, \"input\" -> \"N_2\", \"state\" -> \"Boxes\"|>,\n\"TeXAssistantTemplate\"]\)"},*)
(*LabelStyle->{FontSize->14},*)
(*FrameTicksStyle->Directive[FontFamily->"Latin Modern Math",FontSize->12]*)
(*]]*)


(* ::Subsection:: *)
(*Reduced Form in Output-Space*)


(* ::Subsubsection:: *)
(*Phase Portrait*)


(* ::Input:: *)
(*Show[*)
(*(* ignore error messages from `DisplayPhasePortrait` *)*)
(*Quiet[DisplayPhasePortrait[Ao2d/.{d->0}*)
(*,Tries->300,PlotRange->{{-.05,1.05},{-.05,1.05}}*)
(*,SaddleManifolds->True*)
(*,SaddleManifoldDuration->1000*)
(*,SaddleManifoldPlotPoints->1000*)
(*,NullManifolds->False*)
(*,FrameLabel->{"\!\(\*TemplateBox[<|\"boxes\" -> FormBox[SubscriptBox[StyleBox[\"o\", \"TI\"], \"1\"], TraditionalForm], \"errors\" -> {}, \"input\" -> \"o_1\", \"state\" -> \"Boxes\"|>,\n\"TeXAssistantTemplate\"]\)","\!\(\*TemplateBox[<|\"boxes\" -> FormBox[SubscriptBox[StyleBox[\"o\", \"TI\"], \"2\"], TraditionalForm], \"errors\" -> {}, \"input\" -> \"o_2\", \"state\" -> \"Boxes\"|>,\n\"TeXAssistantTemplate\"]\)"}*)
(*,LabelStyle->{FontSize->14}*)
(*,FrameTicksStyle->Directive[FontFamily->"Latin Modern Math",FontSize->12]*)
(*]]*)
(*,Graphics[{EdgeForm[{Thin,Gray,Dashed}],FaceForm[],Rectangle[{0,0},{1,1}]}]*)
(*,ImageSize->Medium*)
(*]*)


(* ::Subsubsection:: *)
(*Step-Points*)


(* ::Input::Initialization:: *)
p1Phase1=Flatten[Import[FileNameJoin[{NotebookDirectory[],"p1_phase1.txt"}],"Table"]];
p1Phase2=Flatten[Import[FileNameJoin[{NotebookDirectory[],"p1_phase2.txt"}],"Table"]];
p2Phase3=Flatten[Import[FileNameJoin[{NotebookDirectory[],"p2_phase3.txt"}],"Table"]];


(* ::Input::Initialization:: *)
(* Subscript[N, 2] must be approximated due to saturation in output *)
SaddlePoint=EquilibriumPoints[A2d/.{d->0},Tries->300][[2,3]];
Attractor=EquilibriumPoints[A2d/.{d->0},Tries->300][[1,3]];
(* linear approximation to unstable saddle manifold *)
m=(Attractor[[2]]-SaddlePoint[[2]])/(Attractor[[1]]-SaddlePoint[[1]]);
b=SaddlePoint[[2]]-m*SaddlePoint[[1]];
Subscript[\[Sigma], -1][x_]:=Log[x/(1-x)]-\[Theta]1/.A2d[[9]];
StateFromOut[x_]:={Subscript[\[Sigma], -1][x],m Subscript[\[Sigma], -1][x]+b};


(* ::Input::Initialization:: *)
(* manually correcting points to land on transient/manifold *)
p1Phase2Traj=Thread[StateFromOut[p1Phase2]];
p1Phase2Traj = (p1Phase2Traj+{{0,0},{0,-.06},{0,.005},{0,.005}});


(* ::Input:: *)
(*Show[*)
(*DisplayTrajectories[A2d/.{d->0},{{0,0}},1000,*)
(*PlotStyle->{LightGray,Thickness[.003]},Arrows->10,PlotRange->{{10.5,13},{-5.6,-4.4}},FrameLabel->{"\!\(\*TemplateBox[<|\"boxes\" -> FormBox[SubscriptBox[StyleBox[\"N\", \"TI\"], \"1\"], TraditionalForm], \"errors\" -> {}, \"input\" -> \"N_1\", \"state\" -> \"Boxes\"|>,\n\"TeXAssistantTemplate\"]\)","\!\(\*TemplateBox[<|\"boxes\" -> FormBox[SubscriptBox[StyleBox[\"N\", \"TI\"], \"2\"], TraditionalForm], \"errors\" -> {}, \"input\" -> \"N_2\", \"state\" -> \"Boxes\"|>,\n\"TeXAssistantTemplate\"]\)"},*)
(*LabelStyle->{FontSize->14},*)
(*FrameTicksStyle->Directive[FontFamily->"Latin Modern Math",FontSize->12]]*)
(*,Quiet[DisplayPhasePortrait[A2d/.{d->0},Tries->300*)
(*,SaddleManifolds->True*)
(*,SaddleManifoldDuration->1500,*)
(*SaddleManifoldPlotPoints->1000,*)
(*SaddleManifoldArrows->1,*)
(*MaxIterations->1000*)
(*]]*)
(*,ListPlot[p1Phase2Traj,PlotStyle->{Cyan,PointSize[0.025]}]*)
(*,ListPlot[{StateFromOut[p1Phase1[[1]]]-{0,.295}},PlotStyle->{Purple,PointSize[0.02]}]*)
(*,ListPlot[{StateFromOut[p2Phase3[[1]]]},PlotStyle->{Pink,PointSize[0.02]}]*)
(*,Graphics[Text[Framed[Style["\!\(\**)
(*StyleBox[\"low\",\nFontSlant->\"Italic\"]\)",Black,Italic,12,FontFamily->"Latin Modern Roman"],*)
(*Background->None,FrameStyle->None],p1Phase2Traj[[2]]+{.1,.03}]]*)
(*,Graphics[Text[Framed[Style["\!\(\**)
(*StyleBox[\"mid\",\nFontSlant->\"Italic\"]\)",Black,Italic,12,FontFamily->"Latin Modern Roman"],*)
(*Background->None,FrameStyle->None],p1Phase2Traj[[3]]+{.13,.01}]]*)
(*,Graphics[Text[Framed[Style["\!\(\**)
(*StyleBox[\"high\",\nFontSlant->\"Italic\"]\)",Black,Italic,12,FontFamily->"Latin Modern Roman"],*)
(*Background->None,FrameStyle->None],p1Phase2Traj[[4]]+{-.13,-.01}]]*)
(*,ImageSize->Medium*)
(*]*)


(* ::Subsubsection:: *)
(*Verifying Saddle-to-Unstable Reduction*)


(* ::Input:: *)
(*saddle2D=EquilibriumPoints[A2d/.{d->.3},Tries->1000][[1,3]]*)
(*saddle5D=EquilibriumPoints[A/.{d->.3},Tries->1000][[1,3]]*)


(* ::Input:: *)
(*sys5D=A[[10]]/.A[[9]];*)
(*egv5D=Eigenvalues[sys5D/.Thread[A[[3]]->saddle5D]];*)
(*egv5D*)


(* ::Input:: *)
(*egv2D=Eigenvalues[A2d[[10]]]/.A2d[[9]];*)
(*egv2D/.Thread[A2d[[3]]->saddle2D]*)


(* ::Section::Closed:: *)
(*Bifurcation Analysis*)


(* ::Subsection:: *)
(*Fixed Points*)


(* ::Subsubsection:: *)
(*State-Space*)


(* ::Input:: *)
(*DisplayBifurcationDiagram[A2d,d]*)


(* ::Subsubsection:: *)
(*Output-Space*)


(* ::Input:: *)
(*DisplayBifurcationDiagram[Ao2d,d]*)


(* ::Subsection:: *)
(*Bifurcation Diagrams With Limit-Cycle*)


(* ::Subsubsection:: *)
(*Finding the limit cycle*)


(* ::Input:: *)
(*DisplayTrajectories[Ao2d/.{d->0.3},{{.99,.01}},270];*)
(*LC=FindLimitCycle[Ao2d/.{d->.3},Last[First[$Trajectories]],270]*)
(*Show[*)
(*DisplayPhasePortrait[Ao2d/.{d->0.3},Tries->300,SaddleManifolds->True,SaddleManifoldDuration->10000,SaddleManifoldPlotPoints->1000,NullManifolds->True,AccuracyGoal->10,PlotRange->{{-.05,1.05},{-.05,1.05}},*)
(*LimitSets->{LC}]*)
(*]*)
(*LCB=FollowLimitCycleBranch[LC,d,MaxStepSize->0.01];*)


(* ::Subsubsection:: *)
(*Bifurcation Diagram in Output Space*)


(* ::Input:: *)
(*Show[*)
(*DisplayBifurcationDiagram[Ao2d,d,Tries->300,MaxStepSize->0.01,LimitCycleBranches->{LCB},*)
(*BifurcationPointStyle->{PointSize[0.012]},ViewPoint->{1.110396031150695`,1.6853439215169146`,1.0847287772576018`},*)
(*AxesLabel->{"\!\(\*TemplateBox[<|\"boxes\" -> FormBox[StyleBox[\"s\", \"TI\"], TraditionalForm], \"errors\" -> {}, \"input\" -> \"s\", \"state\" -> \"Boxes\"|>,\n\"TeXAssistantTemplate\"]\)","\!\(\*TemplateBox[<|\"boxes\" -> FormBox[SubscriptBox[StyleBox[\"o\", \"TI\"], \"1\"], TraditionalForm], \"errors\" -> {}, \"input\" -> \"o_1\", \"state\" -> \"Boxes\"|>,\n\"TeXAssistantTemplate\"]\)","\!\(\*TemplateBox[<|\"boxes\" -> FormBox[SubscriptBox[StyleBox[\"o\", \"TI\"], \"2\"], TraditionalForm], \"errors\" -> {}, \"input\" -> \"o_2\", \"state\" -> \"Boxes\"|>,\n\"TeXAssistantTemplate\"]\)"},*)
(*AxesStyle->Directive[FontSize->15],*)
(*(* `TicksStyle` option must be added in Dynamica.m; otherwise, remove the following line.*)
(*	An error may still be thrown, even if it works. If this bothers you too much, wrap this `DisplayBifurcationDiagram` function in a `Quiet` function.*)
(**)*)
(*TicksStyle->Directive[FontFamily->"Latin Modern Math",FontSize->12]*)
(*],ImageSize->Medium*)
(*]*)
(*BPs=$BifurcationDiagramBPs;*)


(* ::Input:: *)
(*(* closeup of first bifurcation *)*)
(*Show[*)
(*DisplayBifurcationDiagram[Ao2d,d,Tries->300,BifurcationPointStyle->{PointSize[0.015]},MaxStepSize->0.001,*)
(*AxesLabel->{"\!\(\*TemplateBox[<|\"boxes\" -> FormBox[StyleBox[\"s\", \"TI\"], TraditionalForm], \"errors\" -> {}, \"input\" -> \"s\", \"state\" -> \"Boxes\"|>,\n\"TeXAssistantTemplate\"]\)","\!\(\*TemplateBox[<|\"boxes\" -> FormBox[SubscriptBox[StyleBox[\"o\", \"TI\"], \"1\"], TraditionalForm], \"errors\" -> {}, \"input\" -> \"o_1\", \"state\" -> \"Boxes\"|>,\n\"TeXAssistantTemplate\"]\)","\!\(\*TemplateBox[<|\"boxes\" -> FormBox[SubscriptBox[StyleBox[\"o\", \"TI\"], \"2\"], TraditionalForm], \"errors\" -> {}, \"input\" -> \"o_2\", \"state\" -> \"Boxes\"|>,\n\"TeXAssistantTemplate\"]\)"},*)
(*AxesStyle->Directive[FontSize->15],*)
(*(* `TicksStyle` option must be added in Dynamica.m; otherwise, remove the following line.*)
(*	An error may still be thrown, even if it works. If this bothers you too much, wrap this `DisplayBifurcationDiagram` function in a `Quiet` function.*)
(**)*)
(*TicksStyle->Directive[FontFamily->"Latin Modern Math",FontSize->12],*)
(*Ticks->{{0,0.005},{0.05,0.1,0.15},{0.98,0.99}},*)
(*PlotRange->{{0.0,0.005},{0.05,0.2},{0.97,1.0}},BoxRatios->{1,2,1},*)
(*ViewPoint->{1.4284878636156926`,1.4749015107756438`,1.0169011540038673`}*)
(*]*)
(*,ImageSize->Medium*)
(*]*)


(* ::Input:: *)
(*(* local bifurcations *)*)
(*Print[#]&/@BPs[[;;,1,9,1]];*)


(* ::Subsubsection:: *)
(*Global Bifurcations*)


(* ::Input:: *)
(*Print[#]&/@BPs[[{3,4,5},1,9,1]];*)


(* ::Input:: *)
(**)


(* ::Input:: *)
(*(* phase-portraits near first global-bifurcation *)*)
(*GraphicsRow[*)
(*Module[{},*)
(*DisplayTrajectories[Ao2d/.{d->#},{{.99,.01}},270];*)
(*Quiet[DisplayPhasePortrait[Ao2d/.{d->#},Tries->300,PlotRange->{{-.05,1.05},{-.05,1.05}},FrameLabel->{"\!\(\*SubscriptBox[*)
(*StyleBox[\"o\",\nFontSlant->\"Italic\"], \(1\)]\)","\!\(\*SubscriptBox[*)
(*StyleBox[\"o\",\nFontSlant->\"Italic\"], \(2\)]\)"},*)
(*SaddleManifolds->True,*)
(*SaddleManifoldDuration->1000,*)
(*SaddleManifoldPlotPoints->1000,*)
(*NullManifolds->False,*)
(*LimitSets->{FindLimitCycle[Ao2d/.{d->#},Last[First[$Trajectories]],270]}*)
(*]*)
(*]]&/@Table[BPs[[4,1,9,1,2]]+dd,{dd,{-0.01,0,0.05}}]*)
(*]*)
(**)
(**)


(* ::Subsection:: *)
(*Quasi-static Approximation*)


(* ::Input::Initialization:: *)
traj1=Import[FileNameJoin[{NotebookDirectory[],"naTraj1.txt"}],"Table"];
traj2=Import[FileNameJoin[{NotebookDirectory[],"naTraj2.txt"}],"Table"];


(* ::Input:: *)
(*GraphicsGrid[*)
(*Table[*)
(*Show[*)
(*DisplayBifurcationDiagram[Ao2d,d,Tries->300,LimitCycleBranches->{LCB},StateVariables->{If[i===3,o1,o2]},MaxStepSize->0.01,BifurcationPointStyle->{PointSize[0.01]}],*)
(*ListLinePlot[{Drop[traj,0,{i}]},PlotStyle->Brown]*)
(*]*)
(*,{i,{3,2}},{traj,{traj1,traj2}}*)
(*]*)
(*]*)


(* ::Section::Closed:: *)
(*Interaction Graphs*)


(* ::Subsubsection:: *)
(*Interaction Graph for Receiver in Phase 3*)


(* ::Input:: *)
(*With[{c=Black},*)
(*GraphPlot[{Style[1->4,Red],Style[2->4,Blue],Style[2->5,Red],Style[3->5,Blue],Style[3->1,Red]},*)
(*DirectedEdges->True,EdgeStyle->Thick,*)
(*EdgeLabels->{*)
(*(2->4)->Style["P=1",Italic,FontSize->15,FontColor->c,FontFamily->"Latin Modern Math"],(2->5)->Style["P=2",Italic,FontSize->15,FontColor->c,FontFamily->"Latin Modern Math"]*)
(*},*)
(*VertexStyle->{1->Cyan,2->Cyan,3->Cyan,4->Red,5->Red},VertexSize->.08,*)
(*VertexCoordinates->{1->{1,2},2->{2,1.85},3->{3,2},4->{1,1},5->{3,1}},*)
(*VertexLabels->{*)
(*1->Placed[Style["Low",FontColor->c],Above],2->Placed[Style["Mid",FontColor->c],Above],3->Placed[Style["High",FontColor->c],Above],4->Placed[Style["Slow",FontColor->c],Below],5->Placed[Style["Fast",FontColor->c],Below]*)
(*}*)
(*]*)
(*]*)


(* ::Input:: *)
(**)


(* ::Subsection:: *)
(*Complete Interaction Graph with Redundant Edges*)


(* ::Input:: *)
(*GraphPlot[*)
(*{Style[1->2,Blue],Style[1->4,Red],Style[1->2,Black],Style[1->2,Gray], Style[2->4,Blue],Style[2->3,Gray],Style[2->5,Red],Style[3->4,Gray],Style[3->5,Blue],Style[3->5,Red],Style[4->4,Gray],Style[4->4,Blue],Style[5->1,Red]},*)
(*DirectedEdges->True,EdgeStyle->Thick,*)
(*VertexStyle->{1->Cyan,2->Cyan,3->Cyan,4->Red,5->Red},VertexSize->.08,*)
(*VertexLabels->{*)
(*1->Placed[Style["low",FontColor->Black,FontFamily->"Latin Modern Roman",Italic,FontSize->13],Above],2->Placed[Style["mid",FontColor->Black,FontFamily->"Latin Modern Roman",Italic,FontSize->13],Below],3->Placed[Style["high",FontColor->Black,FontFamily->"Latin Modern Roman",Italic,FontSize->13],Below],4->Placed[Style["slow",FontColor->Black,FontFamily->"Latin Modern Roman",Italic,FontSize->13],{After,Above}],5->Placed[Style["fast",FontColor->Black,FontFamily->"Latin Modern Roman",Italic,FontSize->13],Below]*)
(*}*)
(*]*)


(* ::Input:: *)
(*Export["/home/tom/Documents/Work/Referential Communication/Preprint/Gaul.fig8a.pdf",%53,"PDF"]*)


(* ::Subsubsection:: *)
(*Unravelling the graph for the receiver in \!\(TraditionalForm\`Pu == 1\)*)


(* ::Input:: *)
(*GraphPlot[*)
(*{Style[1->2,Gray],Style[2->3,Gray],Style[3->5,Red],Style[5->1,Red],Style[1->4,Red],Style[4->4,Blue]},*)
(*DirectedEdges->True,EdgeStyle->Thick,*)
(*VertexStyle->{1->Cyan,2->Cyan,3->Cyan,4->Red,5->Red},VertexSize->.08,*)
(*VertexLabels->{*)
(*1->Placed[Style["low",FontColor->Black,FontFamily->"Latin Modern Roman",Italic,FontSize->13],Above],2->Placed[Style["mid",FontColor->Black,FontFamily->"Latin Modern Roman",Italic,FontSize->13],Below],3->Placed[Style["high",FontColor->Black,FontFamily->"Latin Modern Roman",Italic,FontSize->13],Below],4->Placed[Style["slow",FontColor->Black,FontFamily->"Latin Modern Roman",Italic,FontSize->13],{After,Above}],5->Placed[Style["fast",FontColor->Black,FontFamily->"Latin Modern Roman",Italic,FontSize->13],Below]*)
(*}*)
(*]*)


(* ::Subsection:: *)
(*Interaction Graph with Perturbation Classes*)


(* ::Input:: *)
(*GraphPlot[*)
(*{Style[1->2,Orange],Style[1->4,Brown], Style[2->4,Purple],Style[2->3,Magenta],*)
(*Style[2->5,RGBColor[0.9211163550040478, 0.8162447051330659, 0.2590925803004436]],Style[3->4,Pink],Style[3->5,Black],Style[4->4,Cyan],Style[5->1,Green]},*)
(*DirectedEdges->True,EdgeStyle->Thick,*)
(*VertexStyle->{1->Cyan,2->Cyan,3->Cyan,4->Red,5->Red},VertexSize->.08,*)
(*VertexLabels->{*)
(*1->Placed[Style["low",FontColor->Black,FontFamily->"Latin Modern Roman",Italic,FontSize->13],Above],2->Placed[Style["mid",FontColor->Black,FontFamily->"Latin Modern Roman",Italic,FontSize->13],Below],3->Placed[Style["high",FontColor->Black,FontFamily->"Latin Modern Roman",Italic,FontSize->13],Below],4->Placed[Style["slow",FontColor->Black,FontFamily->"Latin Modern Roman",Italic,FontSize->13],{After,Above}],5->Placed[Style["fast",FontColor->Black,FontFamily->"Latin Modern Roman",Italic,FontSize->13],Below]*)
(*}*)
(*]*)


(* ::Input:: *)
(*Export["/home/tom/Documents/Work/Referential Communication/Preprint/Gaul.fig8b.pdf",%55,"PDF"]*)
