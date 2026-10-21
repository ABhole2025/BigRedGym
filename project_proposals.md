---
title: "Assignment 5: Project Proposals"
author: "Abha Bhle ab2792"
date: "October 23, 2026"
documentclass: article
fontsize: 11pt
papersize: letter
geometry: margin=0.85in
pagestyle: empty
header-includes:
  - \usepackage{graphicx}
  - \usepackage{titlesec}
  - \titleformat{\section}{\normalfont\large\bfseries}{}{0pt}{}
  - \titlespacing*{\section}{0pt}{1em}{0.35em}
  - \setlength{\parindent}{0pt}
  - \setlength{\parskip}{0.5em}
  - \raggedbottom
---

# Project A: Sit-to-Stand Transitions

## What do you want to achieve?

Train the Unitree Go2 to transition smoothly between a normal standing posture and a seated posture on its haunches, and return to standing on command.

The goal is to learn both transitions in simulation and attempt sim-to-real transfer onto the Go2. The robot should reach a recognizable sitting posture without falling or losing balance, and be able to return to stable standing.

Stretch goals include considering partial transitions like crouching or maintaining the seated posture for an extended period.

## How will you show it?

- **Transition success rate:** Percentage of attempts that successfully reach the desired posture without failing.
- **Transition time:** Time required to complete standing-to-sitting and sitting-to-standing motions.
- **Stability:** Body pitch, roll, and angular velocity during and after transitions.
- **Smoothness:** Oscillations, abrupt joint movements, and overshoot during transitions.
- **Sim-to-real performance:** Comparison of successful transitions and stability between simulation and hardware.

## Questions and Challenges to Consider

- How should a successful sitting posture be quantified?
- How can the reward encourage sitting on the haunches rather than simply crouching or collapsing?
- Should sitting and standing be learned by a single policy or by separate policies? Is it even within scope to do both, or would it be better to just focus on one?
- How can the robot maintain balance while its center of mass shifts during the transition?
- Are the desired postures and transitions achievable within the Go2's joint and torque limits?


# Project B: Choreographed Leg Tapping

## What do you want to achieve?

Train the Unitree Go2 to perform a predefined sequence of leg taps in a specified order and rhythm while maintaining a stable standing posture.

The minimum goal is to execute and repeatedly perform a four-beat sequence, such as **front-left tap, front-right tap, front-left tap, rest**, at a fixed tempo in simulation, followed by an attempted sim-to-real transfer.

The robot should lift and lower the appropriate foot at the correct point in the sequence without losing balance or significantly changing its position. Additionally, the robot should maintain the designed timing and rhythm.

Stretch goals include longer sequences, more complex tapping patterns, or changing the tempo on command.

## How will you show it?

- **Timing accuracy:** Difference between commanded beat times and actual foot contacts.
- **Tap consistency:** Variation in foot lift height and duration across repeated sequences.
- **Sequence completion:** Number of consecutive sequences completed successfully without losing balance or rhythm.
- **Sim-to-real performance:** Comparison of timing, accuracy, and stability between simulation and hardware.

## Questions and Challenges to Consider

- How should the tapping sequence and timing information be communicated to the policy?
- Should the robot learn the sequence directly, or track a predefined sequence of target foot positions?
- How should rewards balance timing accuracy, foot clearance, and stability?
- How can the robot shift its weight to maintain balance while lifting a leg?
- How accurately can foot-ground contact timing be measured in simulation and on the physical robot?
- Would a fixed tempo be sufficient, or should the policy be trained across a range of tempos?


# Project C: Recovery from External Disturbances

## What do you want to achieve?

Train the Unitree Go2 to recover a stable standing posture after experiencing external disturbances, such as pushes that cause significant body displacement or rotation.

The minimum goal is to recover from moderate disturbances applied while standing still in simulation, followed by an attempted sim-to-real transfer. Successful recovery means returning to an upright, stable posture without falling.

The project will focus on recovering balance rather than recovering from a fully fallen-over position.

Stretch goals include recovery from stronger disturbances, disturbances applied from different directions, or pushes experienced while walking.

## How will you show it?

- **Recovery success rate:** Percentage of disturbances from which the robot returns to stable standing without falling.
- **Recovery time:** Time required for body orientation and motion to return within predefined stability thresholds.
- **Maximum recoverable disturbance:** Largest disturbance magnitude from which the robot can consistently recover.
- **Corrective actions:** Number of recovery steps or foot repositionings required.
- **Sim-to-real performance:** Comparison of recovery success and time between simulation and hardware.

## Questions and Challenges to Consider

- How should disturbances be applied during training: external forces, impulses, or changes in initial conditions?
- What range of disturbance magnitudes should be used to make training challenging but achievable?
- How should stable recovery be defined quantitatively?
- Should the robot be allowed to reposition its feet, or should it attempt recovery without stepping?
- How can the reward encourage effective recovery without producing an excessively stiff standing policy?
- How can simulation disturbances be reproduced safely and consistently on the physical Go2?
