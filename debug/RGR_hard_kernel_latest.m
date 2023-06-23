(* ::Package:: *)

BeginPackage["RGRPackage`"]


(* ::Section:: *)
(*Claim(\:58f0\:660e)*)


RG1loop::usage = "RG1loop[\[Zeta]] returns RGR 1 loop result";

Fix1loop::usage = "Fix1loop[\[Zeta], \[Mu]] returns Fixed order 1 loop result";

RG2loop::usage = "RG2loop[\[Zeta]] returns RGR result with fixed order 2 loop";

Fix2loop::usage = "Fix2loop[\[Zeta], \[Mu]] returns Fixed order 2 loop result";


(* ::Section:: *)
(*Private(\:5b9e\:73b0)*)


Begin["`Private`"]


(* ::Subsection:: *)
(*Constants*)


\[Alpha]smZ = 0.1179;

mZ = 91.19

mb = 4.18 + 0.108 * (\[Alpha]smZ - 0.1182)(* MSbar masses,PDG 10.2.3*)

mc = 1.274 + 2.616 * (\[Alpha]smZ - 0.1182)

CF = 4/3;

CA = 3;

TF = 1/2;


C1 = CF / (2 \[Pi]) * (-2 + (\[Pi]^2) / 12);

C2[nf_] :=
	0.0725 * CF^2 - 0.0840 * CF * CA + 0.1453 * CF * nf * TF;


(* ::Subsection:: *)
(*Running Coupling*)


\[Beta]0[nf_] :=
	-((33 - 2 nf) / (6 \[Pi]));

\[Beta]1[nf_] :=
	-((153 - 19 nf) / (12 \[Pi]^2));

asRun[\[Mu]_, \[CapitalLambda]_, nf_] :=
	-(2 / (\[Beta]0[nf] Log[\[Mu]^2 / \[CapitalLambda]^2])) - (4 \[Beta]1[nf] Log[Log[\[Mu]^2 / \[CapitalLambda]^2]]) / ((
		\[Beta]0[nf]) ^ 3 (Log[\[Mu]^2 / \[CapitalLambda]^2]) ^ 2);


(* Solve \[CapitalLambda]5 *)

\[CapitalLambda]5 = \[CapitalLambda] /. FindRoot[asRun[mZ, \[CapitalLambda], 5] == \[Alpha]smZ, {\[CapitalLambda], 0.25}];

\[Alpha]smb = asRun[mb, \[CapitalLambda], 5] /. {\[CapitalLambda] -> 0.22497225220886866`};

(* Solve \[CapitalLambda]4 *)

\[CapitalLambda]4 = \[CapitalLambda] /. FindRoot[asRun[mb, \[CapitalLambda], 4] == \[Alpha]smb, {\[CapitalLambda], 0.25}];

\[Alpha]smc = asRun[mc, \[CapitalLambda], 4] /. {\[CapitalLambda] -> 0.32167532393289794`};

(* Solve \[CapitalLambda]3 *)

\[CapitalLambda]3 = \[CapitalLambda] /. FindRoot[asRun[mc, \[CapitalLambda], 3] == \[Alpha]smc, {\[CapitalLambda], 0.25}];


(* ::Subsection:: *)
(*Anomalous Dimensions*)


(* Ref: arXiv:2304.14440, note the 4\[Pi] factor difference here. *)

gammaV1 = -3/(2 \[Pi]) * CF;

gammaV2[nf_] := CF / (16 \[Pi]^2) * ( CF * (-3+24*Zeta[2]-48*Zeta[3]) + CA * (-961/27-22*Zeta[2]+52*Zeta[3])+nf*(130/27+4*Zeta[2]) );

gammaV3[nf_]:= CF / (64 \[Pi]^3) * ( CF^2 * ( -36 * Zeta[2] - 136 * Zeta[3] - 288 * Zeta[4] + 64 * Zeta[2] * Zeta[3] + 480 * Zeta[5] - 29 ) 
+ CF*CA * ( 820/3*Zeta[2] - 1688/3*Zeta[3] + 988/3*Zeta[4] - 32 * Zeta[2] * Zeta[3] - 240 * Zeta[5] - 151/2 )
+ CF*nf * ( -52/3*Zeta[2] + 512/9*Zeta[3]-280/3*Zeta[4]+2953/27 )
+ CA^2 * ( -14326/81*Zeta[2] + 7052/9*Zeta[3] - 166 * Zeta[4] - 176/3* Zeta[2] * Zeta[3] - 272 * Zeta[5] - 139345/1458 )
+ CA * nf * ( 5188/81*Zeta[2] - 1928/27 * Zeta[3] + 44 * Zeta[4] - 17318/729 ) + nf^2 * (-40/9*Zeta[2] - 16/27*Zeta[3] + 4834/729) 
  );

gammaJ1 = 3/(8\[Pi]) * CF;

gammaJ2[nf_] := CF / (16 \[Pi]^2) * ( CF * (-5/4 + 8 * Zeta[2]) + CA * (49/12-2*Zeta[2]) - 5/6 * nf );

gammaJ3[nf_]:= CF / (64 \[Pi]^3) * ( CF^2 * ( 37/4-32*Zeta[2]+18 * Zeta[3] + 40 * Zeta[4] )
+ CF * CA * ( 655/72 + 592/9*Zeta[2] - 71/3*Zeta[3] + 8 * Zeta[4] )
+ CF * nf * ( -235/18-112/9*Zeta[2] + 44/3*Zeta[3] )
+ CA^2 * ( -1451/216-130/9* Zeta[2] + 11/3* Zeta[3] + 12 * Zeta[4] )
+ CA * nf * (128/27 + 28/9 * Zeta[2] - 38/3* Zeta[3]) - 35/54* nf^2
);

gammaPsi1 = CF / (4 \[Pi]);

gammaPsi2[nf_] := CF / (16 \[Pi]^2) * ( CA * (49/9-2*Zeta[2]+2*Zeta[3])-10/9*nf );

gammaPsi3[nf_]:=  CF / (64 \[Pi]^3) * ( CA^2 * ( 343/18-304/9*Zeta[2] + 370/9*Zeta[3] + 22 * Zeta[4] + 4 * Zeta[2] * Zeta[3] - 18 * Zeta[5] )
+ CF * nf * ( -55/6+8 * Zeta[3] )
 );
 
cusp1 = CF / \[Pi];

cusp2[nf_] :=
	CF / (4 * \[Pi]^2) * (CA * (67/9 - 2 * Zeta[2]) - 10/9 * nf);

cusp3[nf_] :=
	CF / (16 * \[Pi]^3) * (CA^2 * (245/6 - 268/9 * Zeta[2] + 22/3 * Zeta[3] 
		+ 22 * Zeta[4]) + CF * nf * (-(55/6) + 8 * Zeta[3]) + CA * nf * (-(209
		/27) + 40/9 * Zeta[2] - 28/3 * Zeta[3]) - 4/27 nf^2);
		
(* Ref: arXiv:hep-ph/0607228 *)

cusp4[nf_] := 7849 / ( 256 * \[Pi]^4 ) /; nf==3;
cusp4[nf_] := 4313 / ( 256 * \[Pi]^4 ) /; nf==4;
cusp4[nf_] := 1553 / ( 256 * \[Pi]^4 ) /; nf==5;

(* \|01d6fe_C = 4( \|01d6fe_J - \|01d6fe_\|01d713 ) + \|01d6fe_V *)

gammaC1 = 4 * ( gammaJ1 - gammaPsi1 ) + gammaV1;

gammaC2[nf_] := 4 * ( gammaJ2[nf] - gammaPsi2[nf] ) + gammaV2[nf];

gammaC3[nf_] := 4 * ( gammaJ3[nf] - gammaPsi3[nf] ) + gammaV3[nf];




(* ::Subsection:: *)
(*Integral*)


integral2loop[\[Zeta]_, nf_, \[CapitalLambda]_,start_,end_]:=NIntegrate[1/\[Mu] * ( cusp1 * Log[\[Zeta] / \[Mu]^2] * asRun[\[Mu], \[CapitalLambda], nf] + gammaC1 * asRun[\[Mu], \[CapitalLambda], nf] 
+ cusp2[nf] * Log[\[Zeta] / \[Mu]^2] * asRun[\[Mu], \[CapitalLambda], nf]^2 + gammaC2[nf] * asRun[\[Mu], \[CapitalLambda], nf]^2 
+ cusp3[nf] * Log[\[Zeta] / \[Mu]^2] * asRun[\[Mu], \[CapitalLambda], nf]^3
 )
,{\[Mu], start, end}];

integral3loop[\[Zeta]_, nf_, \[CapitalLambda]_,start_,end_]:=NIntegrate[1/\[Mu] * ( cusp1 * Log[\[Zeta] / \[Mu]^2] * asRun[\[Mu], \[CapitalLambda], nf] + gammaC1 * asRun[\[Mu], \[CapitalLambda], nf] 
+ cusp2[nf] * Log[\[Zeta] / \[Mu]^2] * asRun[\[Mu], \[CapitalLambda], nf]^2 + gammaC2[nf] * asRun[\[Mu], \[CapitalLambda], nf]^2 
+ cusp3[nf] * Log[\[Zeta] / \[Mu]^2] * asRun[\[Mu], \[CapitalLambda], nf]^3 + gammaC3[nf] * asRun[\[Mu], \[CapitalLambda], nf]^3 
+ cusp4[nf] * Log[\[Zeta] / \[Mu]^2] * asRun[\[Mu], \[CapitalLambda], nf]^4
 )
,{\[Mu], start, end}];


(* \[Zeta] < mc^2 *)

int2loopNf3[\[Zeta]_] := integral2loop[\[Zeta], 3, \[CapitalLambda]3,  \[Sqrt]\[Zeta], mc] + integral2loop[\[Zeta], 4, \[CapitalLambda]4,  mc, 2]; (* to 2 GeV *)
int3loopNf3[\[Zeta]_] := integral3loop[\[Zeta], 3, \[CapitalLambda]3,  \[Sqrt]\[Zeta], mc] + integral3loop[\[Zeta], 4, \[CapitalLambda]4,  mc, 2]; (* to 2 GeV *)

(* mc^2 <= \[Zeta] < mb^2 *)

int2loopNf4[\[Zeta]_] := integral2loop[\[Zeta], 4, \[CapitalLambda]4,  \[Sqrt]\[Zeta], 2]; (* to 2 GeV *)
int3loopNf4[\[Zeta]_] := integral3loop[\[Zeta], 4, \[CapitalLambda]4,  \[Sqrt]\[Zeta], 2]; (* to 2 GeV *)

(* mb^2 <= \[Zeta] < mZ^2 *)

int2loopNf5[\[Zeta]_] := integral2loop[\[Zeta], 5, \[CapitalLambda]5,  \[Sqrt]\[Zeta], mb] + integral2loop[\[Zeta], 4, \[CapitalLambda]4,  mb, 2]; (* to 2 GeV *)
int3loopNf5[\[Zeta]_] := integral3loop[\[Zeta], 5, \[CapitalLambda]5,  \[Sqrt]\[Zeta], mb] + integral3loop[\[Zeta], 4, \[CapitalLambda]4,  mb, 2]; (* to 2 GeV *)


(* ::Subsection:: *)
(*Fixed-ordered perturbation*)


(* hnloop is fixed order result without log terms, here \[Mu]^2=\[Zeta] *)

h1loop[\[Zeta]_, nf_, \[CapitalLambda]_] :=
	asRun[Sqrt[\[Zeta]], \[CapitalLambda], nf] * CF / (2 \[Pi]) * (-2 + (\[Pi]^2) / 12)

h2loop[\[Zeta]_, nf_, \[CapitalLambda]_] :=
	asRun[Sqrt[\[Zeta]], \[CapitalLambda], nf] * CF / (2 \[Pi]) * (-2 + (\[Pi]^2) / 12) + asRun[Sqrt[
		\[Zeta]], \[CapitalLambda], nf] ^ 2 * C2[nf]

(* hnloopLog is fixed order result with log terms *)

h1loopLog[\[Zeta]_, \[Mu]_, nf_, \[CapitalLambda]_] :=
	asRun[\[Mu], \[CapitalLambda], nf] * CF / (2 \[Pi]) * (-2 + (\[Pi]^2) / 12 + Log[\[Zeta] / \[Mu]^2] - 1/2
		 * (Log[\[Zeta] / \[Mu]^2]) ^ 2)

h2loopLog[\[Zeta]_, \[Mu]_, nf_, \[CapitalLambda]_] :=
	asRun[Sqrt[\[Zeta]], \[CapitalLambda], nf] * CF / (2 \[Pi]) * (-2 + (\[Pi]^2) / 12 + Log[\[Zeta] / \[Mu]^2]
		 - 1/2 * (Log[\[Zeta] / \[Mu]^2]) ^ 2) + asRun[Sqrt[\[Zeta]], \[CapitalLambda], nf] ^ 2 * (C2[nf] - 
		1/2 * (gammaC2[nf] - \[Beta]0[nf] * C1) * Log[\[Zeta] / \[Mu]^2] - 1/4 * (cusp2[nf] -
		 \[Beta]0[nf] * CF / (2 \[Pi])) * (Log[\[Zeta] / \[Mu]^2]) ^ 2 - \[Beta]0[nf] * CF / (24 \[Pi]) * (
		Log[\[Zeta] / \[Mu]^2]) ^ 3)


(* ::Subsection:: *)
(*Output*)


(* RGR 1 loop *)

RG1loop[\[Zeta]_] :=
	int2loopNf3[\[Zeta]] + h1loop[\[Zeta], 3, \[CapitalLambda]3] /; \[Zeta] < mc^2

RG1loop[\[Zeta]_] :=
	int2loopNf4[\[Zeta]] + h1loop[\[Zeta], 4, \[CapitalLambda]4] /; mc^2 <= \[Zeta] < mb^2

RG1loop[\[Zeta]_] :=
	int2loopNf5[\[Zeta]] + h1loop[\[Zeta], 5, \[CapitalLambda]5] /; mb^2 <= \[Zeta] < mZ^2

(* Fixed order 1 loop *)

Fix1loop[\[Zeta]_, \[Mu]_] :=
	h1loopLog[\[Zeta], \[Mu], 3, \[CapitalLambda]3] /; \[Mu] < mc^2

Fix1loop[\[Zeta]_, \[Mu]_] :=
	h1loopLog[\[Zeta], \[Mu], 4, \[CapitalLambda]4] /; mc^2 <= \[Mu] < mb^2

Fix1loop[\[Zeta]_, \[Mu]_] :=
	h1loopLog[\[Zeta], \[Mu], 5, \[CapitalLambda]5] /; mb^2 <= \[Mu] < mZ^2

(* RGR with 2 loop fixed order *)

RG2loop[\[Zeta]_] :=
	int3loopNf3[\[Zeta]] + h2loop[\[Zeta], 3, \[CapitalLambda]3] /; \[Zeta] < mc^2

RG2loop[\[Zeta]_] :=
	int3loopNf4[\[Zeta]] + h2loop[\[Zeta], 4, \[CapitalLambda]4] /; mc^2 <= \[Zeta] < mb^2

RG2loop[\[Zeta]_] :=
	int3loopNf5[\[Zeta]] + h2loop[\[Zeta], 5, \[CapitalLambda]5] /; mb^2 <= \[Zeta] < mZ^2

(* Fixed order 1 loop *)

Fix2loop[\[Zeta]_, \[Mu]_] :=
	h2loopLog[\[Zeta], \[Mu], 3, \[CapitalLambda]3] /; \[Mu] < mc^2

Fix2loop[\[Zeta]_, \[Mu]_] :=
	h2loopLog[\[Zeta], \[Mu], 4, \[CapitalLambda]4] /; mc^2 <= \[Mu] < mb^2

Fix2loop[\[Zeta]_, \[Mu]_] :=
	h1loopLog[\[Zeta], \[Mu], 5, \[CapitalLambda]5] /; mb^2 <= \[Mu] < mZ^2


End[](*`Private`*)


EndPackage[]
