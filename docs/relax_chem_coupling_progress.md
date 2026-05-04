\# Relaxation + Chemistry Coupling Progress



Date range: April 27 – May 4, 2026  

Branch: `phasechange\_local\_dev`



\## Goal



Develop and validate infrastructure for coupling phase-change/vaporization with gas-phase chemistry in MFC.



Target physical sequence:



1\. Shock interacts with a liquid droplet.

2\. Liquid vaporizes.

3\. Vaporized fuel enters the gas-phase species field.

4\. Fuel mixes with oxidizer.

5\. Gas-phase chemistry consumes fuel/oxidizer and forms products.



This branch is not yet a final physical dodecane burning-droplet model. It establishes and validates the coupling infrastructure.



\---



\## Architecture



For live vaporization, the intended 3-fluid layout is:



\- Fluid 1: liquid fuel

\- Fluid 2: fuel vapor

\- Fluid 3: oxidizer/carrier gas



The phase-change module currently assumes:



\- `lp = 1`

\- `vp = 2`



So when `relax = T`, fluid 2 must remain the vapor phase.



For surrogate burning-wake prototypes with `relax = F`, a simpler 2-fluid setup is acceptable:



\- Fluid 1: liquid droplet surrogate

\- Fluid 2: gas mixture carrying H2/O2/N2/H2O species



In that case, fuel vapor is represented as a gas species, not as a separate vapor fluid.



\---



\## Main Code Changes



\### Chemistry gas density



Added support for selecting the gas density used by chemistry.



Existing fallback:



\- `chem\_gas\_fluid\_id`



New combined-gas support:



\- `chem\_gas\_num\_fluids`

\- `chem\_gas\_fluid\_ids(:)`



This enables:



```text

rho\_g = rho\_vapor + rho\_air

