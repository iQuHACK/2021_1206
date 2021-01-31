# Optimizing COVID-19 Vaccine Distribution Using Quantum Annealing

Abstract
---
The rate at which people are being vaccinated is a lot slower than initially predicted by government officials. This may be due to a combination of factors such as delays in shipments to distribution centers, long travel distances for patients, long wait times for the vaccine, and leftover vaccinations expiring. A lack of a government plan for vaccine distribution has the potential to increase the number of deaths in the US and also weaken the already strained healthcare system via more hospitalizations. Currently, COVID-19 vaccine distribution is in its early phases, providing doses to those over the age of 65 and to essential healthcare workers. As the vaccine becomes more readily available to the public, states face large logistical issues of where to add additional sites to handle the necessary demand. Adding too many sites leads to costs for special storage equipmemt, healthcare workers to administer vaccines, and incur transportation costs to transport the vaccines to the sites themselves. On the contrary, not having enough sites leads to large waiting times and slow rollout of the vaccine to the general public. We aim to frame this as an optimization problem to determine the effect of number of vaccine distribution sites on the average distance individuals have to travel to receive a vaccine using D-Wave's binary quantum model (BQM) solver. 



Summary
---
Our project focuses specifically on Texas and the population of age 65+ citizens grouped by county, but this approach can be adapted to reflect national level distribution of vaccines for the general public (see more below). This data was scraped from: 

We implement an algorithm based off Placement of Charging Stations (Goliber 2020) to determine where to place N new distribution sites with the following constraints:

1. Minimum average distance to county center points

2. Prefer counties with higher population
3. N desired number of new distribution centers 

 


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
