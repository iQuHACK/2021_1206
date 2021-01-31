# Optimizing COVID-19 Vaccine Distribution Using Quantum Annealing

Abstract
---
Quantum annealing hardware has demonstrated the potential to solve discrete optimization problems by using qubits to model physical energy states. Using DWave’s annealing technology, we aim to gain insight into bettering process of vaccine distribution in order to maximize the number of people vaccinated in the shortest possible time. 



Motivation
---
The rate at which people are being vaccinated is a lot slower than initially predicted by government officials. This may be due to a combination of factors such as delays in shipments to distribution centers, long travel distances for patients, long wait times for the vaccine, and leftover vaccinations expiring. A lack of a government plan for vaccine distribution has the potential to increase the number of deaths in the US and also weaken the already strained healthcare system via more hospitalizations. Therefore, we aim to find the best allocation of individuals to COVID-19 vaccine distribution centers that minimizes the distance individuals must travel to reach the site while simultaneously maximizing the number of individuals that receive the vaccination. 



Summary
---
Our project focuses specifically on Texas and the population of age 65+ citizens grouped by county, but this approach can be adapted to reflect national level distribution of vaccines for the general public (see more below). Our optimization problem takes into account the following statistics

* the number of individuals allocated to each site
* the number of existing distribution centers
* the desired number of new distribution centers 

Using the above data, we attempt to model the situation where there are more patients than distribution sites and map individual patients to their nearest distribution site. We do that by:

1) minimizing the average distance from counties to distribution centers
- maybe talk a lil more about how

2) maximizing the distance between distribution centers
- a lil more about how

Some limitations of this project include the assumptions

* individuals will have access to transportation
* distribution centers have unlimited vaccine capacity
* populations are concentrated at the geographical center of the counties

We only look at the allocation of individuals to a specific distribution site, and we do not take into account dosage timings. We also do not take into account the number of vaccines available at a distribution center. However, these constraints can be modeled on a more complex optimization problem that we hope to explore in the future!



Analysis
---
In studying this data, we can gain insight into vaccine and other resource allocations. By strategically "inserting" distribution centers to minimize the average distance patients would have to travel, we identify the locations that should should be the most prioritized in vaccine and resource allocations. (idk what to say BWAHAHAHAH I feel like our project objective has been blurred)



Future Work
---
Currently, we are assigning individuals to distribution sites and then allocating vaccines. Once vaccinations become available for the general public, there will be more a lot more demand compared to supply. In this case, (WIP)
* 1) the number of vaccines available from Pfizer and Moderna



iQuHACK 2020 Team 1206
---
Mindy Long
Shreya Karpoor
Elaine Pham
