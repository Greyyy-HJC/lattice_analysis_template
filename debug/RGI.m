(* ::Package:: *)

BeginPackage["RGIPackage`"]


(* ::Section:: *)
(*\:58f0\:660e*)


RGInt::usage = "RGInt[\[Zeta]] returns RGI result"; 


(* ::Section:: *)
(*Private(\:5b9e\:73b0)*)


Begin["`Private`"]

\[Beta]0[nf_] :=
    -((33 - 2 nf) / (6 \[Pi])); 

\[Beta]1[nf_] :=
    -((153 - 19 nf) / (12 \[Pi]^2)); 

asRun[\[Mu]_, \[CapitalLambda]_, nf_] :=
    -(2 / (\[Beta]0[nf] Log[\[Mu]^2 / \[CapitalLambda]^2])) - (4 \[Beta]1[nf] Log[Log[\[Mu]^2 / \[CapitalLambda]^2]]) /
         ((\[Beta]0[nf])^3 (Log[\[Mu]^2 / \[CapitalLambda]^2])^2); 

\[Alpha]smZ = 0.1179; 

mZ = 91.19; 

mb = 4.18 + 0.108 * (\[Alpha]smZ - 0.1182)(* MSbar masses,PDG 10.2.3*)

mc = 1.274 + 2.616 * (\[Alpha]smZ - 0.1182)

(* Solve \[CapitalLambda]5 *)

\[CapitalLambda]5 = \[CapitalLambda] /. FindRoot[asRun[mZ, \[CapitalLambda], 5] == \[Alpha]smZ, {\[CapitalLambda], 0.25}]

\[Alpha]smb = asRun[mb, \[CapitalLambda], 5] /. {\[CapitalLambda] -> 0.22497225220886866`}

(* Solve \[CapitalLambda]4 *)

\[CapitalLambda]4 = \[CapitalLambda] /. FindRoot[asRun[mb, \[CapitalLambda], 4] == \[Alpha]smb, {\[CapitalLambda], 0.25}]

\[Alpha]smc = asRun[mc, \[CapitalLambda], 4] /. {\[CapitalLambda] -> 0.32167532393289794`}

(* Solve \[CapitalLambda]3 *)

\[CapitalLambda]3 = \[CapitalLambda] /. FindRoot[asRun[mc, \[CapitalLambda], 3] == \[Alpha]smc, {\[CapitalLambda], 0.25}]

cusp1 = CF / \[Pi]; 

cusp2[nf_] :=
    CA * CF * (67 / (36 * \[Pi]^2) - 1 / 12) - (5 * CF * nf) / (18 \[Pi]^2); 

gammac1 = 2 * ((3 CF) / (4 \[Pi])) - CF / \[Pi] + 2 * ((-3 CF) / (4 \[Pi]));  (* changed *)

gammac2[nf_]:=1/(16\[Pi]^2)*(CF*CA*(-(1108/27)-(11 \[Pi]^2)/3+44 Zeta[3])+CF^2*(-48*Zeta[3]+(28 \[Pi]^2)/3-8)+CF*nf*(2/27*(80+9 \[Pi]^2))); (* changed as Yu-Shan suggested *)

CF = 4 / 3; 

CA = 3; 

(* \[Zeta] < mc^2 *)

int1[\[Zeta]_] :=
    NIntegrate[1 / mu * (cusp1 * Log[\[Zeta] / mu^2] * asRun[mu, \[CapitalLambda], nf] + gammac1
         * asRun[mu, \[CapitalLambda], nf] + cusp2[nf] * Log[\[Zeta] / mu^2] * (asRun[mu, \[CapitalLambda], nf]^2
        ) + gammac2[nf] * (asRun[mu, \[CapitalLambda], nf]^2)) /. {nf -> 3, \[CapitalLambda] -> \[CapitalLambda]3}, {mu,
         \[Sqrt]\[Zeta], mc}] + NIntegrate[1 / mu * (cusp1 * Log[\[Zeta] / mu^2] * asRun[mu, \[CapitalLambda],
         nf] + gammac1 * asRun[mu, \[CapitalLambda], nf] + cusp2[nf] * Log[\[Zeta] / mu^2] * (asRun[
        mu, \[CapitalLambda], nf]^2) + gammac2[nf] * (asRun[mu, \[CapitalLambda], nf]^2)) /. {nf -> 4, \[CapitalLambda]
         -> \[CapitalLambda]4}, {mu, mc, 2}]

(* mc^2 <= \[Zeta] < mb^2 *)

int2[\[Zeta]_] :=
    NIntegrate[1 / mu * (cusp1 * Log[\[Zeta] / mu^2] * asRun[mu, \[CapitalLambda], nf] + gammac1
         * asRun[mu, \[CapitalLambda], nf] + cusp2[nf] * Log[\[Zeta] / mu^2] * (asRun[mu, \[CapitalLambda], nf]^2
        ) + gammac2[nf] * (asRun[mu, \[CapitalLambda], nf]^2)) /. {nf -> 4, \[CapitalLambda] -> \[CapitalLambda]4}, {mu,
         \[Sqrt]\[Zeta], 2}]

(* mb^2 <= \[Zeta] < mZ^2 *)

int3[\[Zeta]_] :=
    NIntegrate[1 / mu * (cusp1 * Log[\[Zeta] / mu^2] * asRun[mu, \[CapitalLambda], nf] + gammac1
         * asRun[mu, \[CapitalLambda], nf] + cusp2[nf] * Log[\[Zeta] / mu^2] * (asRun[mu, \[CapitalLambda], nf]^2
        ) + gammac2[nf] * (asRun[mu, \[CapitalLambda], nf]^2)) /. {nf -> 5, \[CapitalLambda] -> \[CapitalLambda]5}, {mu,
         \[Sqrt]\[Zeta], mb}] + NIntegrate[1 / mu * (cusp1 * Log[\[Zeta] / mu^2] * asRun[mu, \[CapitalLambda],
         nf] + gammac1 * asRun[mu, \[CapitalLambda], nf] + cusp2[nf] * Log[\[Zeta] / mu^2] * (asRun[
        mu, \[CapitalLambda], nf]^2) + gammac2[nf] * (asRun[mu, \[CapitalLambda], nf]^2)) /. {nf -> 4, \[CapitalLambda]
         -> \[CapitalLambda]4}, {mu, mb, 2}]

h01[\[Zeta]_] :=
    asRun[Sqrt[\[Zeta]], \[CapitalLambda], nf] * CF / (2 \[Pi]) * (-2 + (\[Pi]^2) / 12) /. {nf -> 
        3, \[CapitalLambda] -> \[CapitalLambda]3}

h02[\[Zeta]_] :=
    asRun[Sqrt[\[Zeta]], \[CapitalLambda], nf] * CF / (2 \[Pi]) * (-2 + (\[Pi]^2) / 12) /. {nf -> 
        4, \[CapitalLambda] -> \[CapitalLambda]4}

h03[\[Zeta]_] :=
    asRun[Sqrt[\[Zeta]], \[CapitalLambda], nf] * CF / (2 \[Pi]) * (-2 + (\[Pi]^2) / 12) /. {nf -> 
        5, \[CapitalLambda] -> \[CapitalLambda]5}

RGInt[\[Zeta]_] :=
    int1[\[Zeta]] + h01[\[Zeta]] /; \[Zeta] < mc^2

RGInt[\[Zeta]_] :=
    int2[\[Zeta]] + h02[\[Zeta]] /; mc^2 <= \[Zeta] < mb^2

RGInt[\[Zeta]_] :=
    int3[\[Zeta]] + h03[\[Zeta]] /; mb^2 <= \[Zeta] < mZ^2


End[](*`Private`*)


EndPackage[]
