# Optimizing COVID-19 Vaccine Distribution Using Quantum Annealing

Abstract
---
Quantum annealing hardware has demonstrated the potential to solve discrete optimization problems by using qubits to model physical energy states. Using DWave’s annealing technology, we aim to gain solve problems relating to vaccine distribution. Our project's goal was to maximize the number of people vaccinated in the shortest possible time by reducing the distance they traveled. 


Motivation
---
The rate at which people are being vaccinated is a lot slower than initially predicted by government officials. This may be due to a combination of factors such as delays in shipments to distribution centers, long travel distances for patients, long wait times for the vaccine, and leftover vaccinations expiring. A lack of a government plan for vaccine distribution has the potential to increase the number of deaths in the US and also weaken the already strained healthcare system via more hospitalizations. Therefore, we aim to find the best sites to place vaccine distribution center, minimizing the distance individuals must travel to reach the site while simultaneously maximizing the number of individuals that receive the vaccination.



Summary
---
Our project focuses specifically on Texas and the population of age 65+ citizens at a county level, but this approach can be adapted to reflect national level distribution of vaccines for the general public (see more below). Texas offered a favorabe case study with its mix of urban and rural centers. Its public health data was also readily available online. Our optimization problem takes into account the following factors

* number of at-risk population of a county
* the number of existing distribution centers
* the desired number of new distribution centers 

Using the above data, we attempt to model the situation where there are more patients than distribution sites and map individual patients to their nearest distribution site. We do that by:

1) Minimizing the average distance from counties to distribution centers
We defined the average distance between individuals and a distribution center as 
![equation](https://latex.codecogs.com/gif.latex?%5Csum_%7Bi%20%3D%201%7D%5E%7B%7CD%7C%7D%20%5Cfrac%7B1%7D%7B%7CP%7C%7Dd_i)  

where *P* is the set of all counties and *D* is the set of all distances between each county and the distribution site.

Thus our constraint couble be seen as
![equation](https://latex.codecogs.com/gif.latex?%5Ctext%7Bmin%7D%5Csum_%7Bj%3D1%7D%5E%7B%7CE%7C%7D%5Csum_%7Bi%20%3D%201%7D%5E%7B%7CD%7C%7D%20%5Cfrac%7B1%7D%7B%7CP%7C%7Dd%7B_i%2Cj%7D)  

where *E* is the set of all distribution centers and *d_i,j* is the distance between distribution center *j* and county *i*.

2) Maximizing the distance between distribution centers
We would like the distribution centers to be spaced throughout the state. We did this by maximizing the distance 
as such:
![equation](https://latex.codecogs.com/gif.latex?%5Ctext%7Bmax%7D%20%5Csum_%7Bi%7D%5E%7B%7CV%7C%7D%5Cfrac%7B1%7D%7B%7CV%7C%7D%5Csum_%7Bi%20%5Cneq%20j%7D%5E%7B%7CV%7C%7D%20%28a_%7Bi_x%7D%20-%20a_%7Bj_x%7D%29%5E2%20%28a_%7Bi_y%7D%20-%20a_%7Bj_y%7D%29%5E2)


Results
--




Analysis
---
In studying this data, we can gain insight into vaccine and other resource allocations. By strategically "inserting" distribution centers to minimize the average distance patients would have to travel, we identify the locations that should should be the most prioritized in vaccine and resource allocations. (idk what to say BWAHAHAHAH I feel like our project objective has been blurred)


Limitations
--
Some limitations of this project include the assumptions

* individuals will have access to transportation
* distribution centers have unlimited vaccine capacity
* populations are concentrated at the geographical center of the counties

We only look at the allocation of individuals to a specific distribution site, and we do not take into account dosage timings. We also do not take into account the number of vaccines available at a distribution center. However, these constraints can be modeled on a more complex optimization problem that we hope to explore in the future!


Future Work
---
We believe that a better model of this problem would seek to minimize the distance between a county and its closest vaccine distribution center. We also would recommend iterating this process at a smaller scale, like zip code, to better serve local communities. We defined at-risk populations as only those over the age of 65 but we can expand that definition and include individuals with a history of chronic and respiratory diseases.



iQuHACK 2020 Team 1206
---
Mindy Long  
Shreya Karpoor  
Elaine Pham  
