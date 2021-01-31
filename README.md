# Optimizing COVID-19 Vaccine Distribution

## Abstract
Quantum annealing finds the global minimum of an objective function over a set of states. In this project, we model COVID-19 vaccine distribution in the state of Texas a graph and seek to minimize the average distance an individual must travel to a vaccine center.
=======
# Speeding Up the Distribution of Vaccines

Abstract
---
Quantum annealing hardware has demonstrated the potential to solve discrete optimization problems by using qubits to model physical energy states. Using DWave’s annealing technology, we aim to gain insight into the process of distributing vaccines which maximizes the number of people vaccinated in the shortest possible time. 



Motivation
---
The rate at which people are being vaccinated is a lot slower than initially predicted by government officials. This may be due to a combination of factors such as delays in shipping vaccines to distribution centers, long distances that patients must travel to receive a vaccine, long wait times for the vaccine, and leftover vaccinations expiring. A lack of a government plan for vaccine distribution has the potential to increase the number of deaths in the US and also weaken the already strained healthcare system via more hospitalizations. Therefore, we aim to find the best allocation of individuals to COVID-19 vaccine distribution centers that minimizes the distance individuals must travel to reach the site while simultaneously maximizing the number of individuals that receive the vaccination. 



Summary
---
Our project focuses specifically on Texas and the population of 65+ citizens grouped by county, but this approach can be adapted to reflect national level distribution of vaccines (see more below). Our optimization problem takes into account the following statistics

* 1) the number of vaccines available from Pfizer and Moderna
* 2) the number of individuals allocated to each site

Some limitations of this project include the assumptions

* individuals will have access to transportation
* distribution centers have unlimited vaccine capacity
* populations are concentrated at the geographical center of the counties

We only look at the allocation of individuals to a specific distribution site, and we do not take into account dosage timings.



Project Formulation
---
Constraint 1: The average distance each individual would have to travel to a distribution center would be the sum of the distance from each county C_i to each distribution center D_i multiplied by the population of the county.

We want to minimize this distance. 

Once we map each individual to a distribution site, we use that number to allocate enough vaccinations for that distribution site. 

Should prolly think about mobile clinics


Future Work
---
Currently, we are assigning individuals to distribution sites and then allocating vaccines. Once vaccinations become available for the general public, there will be more a lot more demand compared to supply. In this case, 



iQuHACK 2020 Team 1206
---
Mindy Long
Shreya Karpoor
Elaine Pham
