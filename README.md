# Resource-Allocation-in-Cloud-Computing-using-PSO-and-Fuzzy-Logic
<p>
  Hello everyone

Cloud computing has become a viable option for many organizations due to its flexibility and scalability in providing virtualized resources via the Internet. It offers the possibility of hosting pervasive applications in the consumer, scientific, and business domains utilizing a pay-as-you-go model. This makes cloud computing a cost-effective solution for businesses as it eliminates the need for large investments in hardware and software infrastructure. Furthermore, cloud computing enables organizations to quickly and easily scale their services to meet the demands of their customers. Resource allocation is a major challenge in cloud computing. It is known as the NP-hard problem and can be solved using meth-heuristic algorithms. This study optimizes resource allocation using the Particle Swarm Optimization (PSO) algorithm and fuzzy logic system developed under the proposed time and cost models in the cloud computing environment. Receiving, processing, and waiting time are included in the time model. The cost model incorporates processing and receiving costs. </p>
<p>In this project, i have implemented a code to improve resource allocation using fuzzy Logic and Mopso algorithms.</p>


<p>

# Step 1: Import required libraries
```
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm
import random
```
# Step 2: Define parameters for generating data
```
num_users = 20   # Number of users
num_providers = 5   # Number of providers

# Resource Attributes according to the paper
MIPS_range = (220, 1000)     # Processing power (Million Instructions Per Second)
Memory_options = [256, 512, 1024, 2048]  # Memory in MB
Accumulator_range = (1500, 40000)  # Accumulator in MB
Bandwidth_range = (120, 1000)  # Bandwidth in bits per second
Cost_range = (0.012, 0.1046)  # Cost per Million Instructions
```
# Step 3: Generate Random Data for Users' Requests
```
users = []
for _ in range(num_users):
    user = {
        'MIPS': np.random.randint(MIPS_range[0], MIPS_range[1]),
        'Memory': np.random.choice(Memory_options),
        'Accumulator': np.random.randint(Accumulator_range[0], Accumulator_range[1]),
        'Bandwidth': np.random.randint(Bandwidth_range[0], Bandwidth_range[1]),
        'Cost': round(np.random.uniform(Cost_range[0], Cost_range[1]), 4)
    }
    users.append(user)
```
# Step 4: Generate Random Data for Providers' Resources
```
providers = []
for _ in range(num_providers):
    provider = {
        'MIPS': np.random.randint(MIPS_range[0], MIPS_range[1]),
        'Memory': np.random.choice(Memory_options),
        'Accumulator': np.random.randint(Accumulator_range[0], Accumulator_range[1]),
        'Bandwidth': np.random.randint(Bandwidth_range[0], Bandwidth_range[1]),
        'Cost': round(np.random.uniform(Cost_range[0], Cost_range[1]), 4)
    }
    providers.append(provider)
```
# Step 5: Visualize Data Distributions
```
attributes = ['MIPS', 'Memory', 'Accumulator', 'Bandwidth', 'Cost']
for attr in attributes:
    user_vals = [user[attr] for user in users]
    provider_vals = [provider[attr] for provider in providers]
    plt.figure(figsize=(10, 5))
    sns.histplot(user_vals, color='blue', label='Users', kde=True)
    sns.histplot(provider_vals, color='green', label='Providers', kde=True)
    plt.title(f'Distribution of {attr}')
    plt.legend()
    plt.show()
```
# Step 6: PSO Parameters
```
population_size = 75
max_iterations = 100
w = 0.4  # Inertia Weight
c1 = 1  # Personal Learning Coefficient
c2 = 1  # Social Learning Coefficient
mutation_rate = 0.5  # Mutation Rate
```
# Step 7: Initialize Particles
```
particles = []
for _ in range(population_size):
    particle = {
        'position': np.random.uniform(0, 1, len(attributes)),
        'velocity': np.zeros(len(attributes)),
        'pbest_position': None,
        'pbest_value': float('inf'),
        'fitness': float('inf'),
        'objectives': None  # Store multiple objectives
    }
    particles.append(particle)
gbest_value = float('inf')  # Set initial global best value to infinity
gbest_position = None
```
# Step 8: Fitness Function
```
# Objective: Minimize Cost while Maximizing Resource Utilization

def fitness_function(position):
    total_cost = 0
    total_utilization = 0
    for i, attr in enumerate(attributes):
        total_cost += position[i] * Cost_range[1]  # Approximate cost
        total_utilization += position[i]  # Utilization as percentage
    return total_cost, total_utilization
    # Fuzzy Inference and Defuzzification
    fuzzy_cost = defuzzification(fuzzy_inference(total_cost, total_utilization))
    
    return fuzzy_cost, total_utilization

# Fuzzy Inference and Aggregation
def fuzzy_inference(cost, utilization):
    memberships = {
        'Cost': {'Low': cost_low(cost), 'Medium': cost_medium(cost), 'High': cost_high(cost)},
        'Utilization': {'Low': util_low(utilization), 'Medium': util_medium(utilization), 'High': util_high(utilization)}
    }
    rule_activations = []
    for cost_label, util_label, output_label in rules:
        activation = min(memberships['Cost'][cost_label], memberships['Utilization'][util_label])
        rule_activations.append((activation, output_label))
    return rule_activations

# Defuzzification
def defuzzification(rule_activations):
    numerator = sum(activation * output_values[label] for activation, label in rule_activations)
    denominator = sum(activation for activation, _ in rule_activations)
    return numerator / denominator if denominator != 0 else 0

# Fuzzy Logic Setup
# Define Triangular Membership Functions for Cost
cost_low = lambda x: max(min((0.05 - x) / 0.03, 1, (x - 0.012) / 0.03), 0)
cost_medium = lambda x: max(min((0.08 - x) / 0.03, (x - 0.05) / 0.03), 0)
cost_high = lambda x: max(min((0.1046 - x) / 0.02, (x - 0.08) / 0.02), 0)

# Define Triangular Membership Functions for Utilization
util_low = lambda x: max(min((0.4 - x) / 0.2, 1, (x - 0.2) / 0.2), 0)
util_medium = lambda x: max(min((0.7 - x) / 0.2, (x - 0.4) / 0.2), 0)
util_high = lambda x: max(min((1.0 - x) / 0.3, (x - 0.7) / 0.3), 0)

# Define Fuzzy Rules
rules = [
    ('Low', 'High', 'Very Good'),
    ('Low', 'Medium', 'Good'),
    ('Low', 'Low', 'Poor'),
    ('Medium', 'High', 'Good'),
    ('Medium', 'Medium', 'Average'),
    ('Medium', 'Low', 'Poor'),
    ('High', 'High', 'Average'),
    ('High', 'Medium', 'Poor'),
    ('High', 'Low', 'Very Poor')
]

# Define Output Membership Values
output_values = {
    'Very Good': 0.9,
    'Good': 0.75,
    'Average': 0.5,
    'Poor': 0.25,
    'Very Poor': 0.1
}
```
# Step 9: MOPSO Setup
```
repository = []  # External Repository for Pareto Optimal Solutions
archive_size = 50  # Maximum size of the repository

# The rest of the MOPSO and PSO code will continue as before.
# Just make sure to call the fitness function properly as shown in Step 8.
```
# Step 10: Particle Swarm Optimization (PSO)
```
def update_velocity(particle, gbest_position, w, c1, c2):
    r1, r2 = np.random.rand(2)
    cognitive_component = c1 * r1 * (particle['pbest_position'] - particle['position'])
    social_component = c2 * r2 * (gbest_position - particle['position'])
    new_velocity = w * particle['velocity'] + cognitive_component + social_component
    return new_velocity

def update_position(particle):
    new_position = particle['position'] + particle['velocity']
    new_position = np.clip(new_position, 0, 1)  # Ensure position is within bounds
    return new_position
```
# Step 11: MOPSO Algorithm
```
def run_mopso(particles, max_iterations, w, c1, c2):
    gbest_position = None
    gbest_value = float('inf')
    
    for iteration in range(max_iterations):
        for particle in particles:
            # Calculate the fitness of the particle
            fuzzy_cost, total_utilization = fitness_function(particle['position'])
            particle['fitness'] = fuzzy_cost
            particle['objectives'] = (fuzzy_cost, total_utilization)
            
            # Update personal best
            if particle['fitness'] < particle['pbest_value']:
                particle['pbest_position'] = particle['position']
                particle['pbest_value'] = particle['fitness']
        
        # Update global best
        for particle in particles:
            if particle['fitness'] < gbest_value:
                gbest_position = particle['position']
                gbest_value = particle['fitness']
        
        # Update velocity and position of each particle
        for particle in particles:
            particle['velocity'] = update_velocity(particle, gbest_position, w, c1, c2)
            particle['position'] = update_position(particle)

        # Print the best solution so far
        print(f"Iteration {iteration + 1}/{max_iterations}, Best Fitness: {gbest_value}")
    
    return gbest_position, gbest_value
```
# Step 12: Run the MOPSO algorithm
```
best_position, best_fitness = run_mopso(particles, max_iterations, w, c1, c2)
```
# Step 13: Display Results
```
print("Best Position (Resource Allocation):", best_position)
print("Best Fitness (Cost and Utilization):", best_fitness)
```
# Step 14: Visualizing the Results
```
# Plot the best fitness over iterations
iterations = np.arange(max_iterations)
fitness_values = []

# Store fitness values for visualization
for iteration in range(max_iterations):
    fitness_values.append(gbest_value)

plt.plot(iterations, fitness_values, label='Best Fitness')
plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.title('PSO Optimization Convergence')
plt.legend()
plt.show()

```
# Step 15: Fuzzy Logic Inference - Apply to Best Solution
```
# Define Triangular Membership Functions for Cost
def cost_low(x):
    return max(min((0.05 - x) / 0.03, 1, (x - 0.012) / 0.03), 0)

def cost_medium(x):
    return max(min((0.08 - x) / 0.03, (x - 0.05) / 0.03), 0)

def cost_high(x):
    return max(min((0.1046 - x) / 0.02, (x - 0.08) / 0.02), 0)

# Define Triangular Membership Functions for Utilization
def util_low(x):
    return max(min((0.4 - x) / 0.2, 1, (x - 0.2) / 0.2), 0)

def util_medium(x):
    return max(min((0.7 - x) / 0.2, (x - 0.4) / 0.2), 0)

def util_high(x):
    return max(min((1.0 - x) / 0.3, (x - 0.7) / 0.3), 0)
```
# Step 16: Fuzzy Inference for Best Solution
```
def fuzzy_inference(cost, utilization):
    memberships = {
        'Cost': {'Low': cost_low(cost), 'Medium': cost_medium(cost), 'High': cost_high(cost)},
        'Utilization': {'Low': util_low(utilization), 'Medium': util_medium(utilization), 'High': util_high(utilization)}
    }
    rule_activations = []
    for cost_label, util_label, output_label in rules:
        activation = min(memberships['Cost'][cost_label], memberships['Utilization'][util_label])
        rule_activations.append((activation, output_label))
    return rule_activations

# Fuzzy rules based on the previous definitions
rules = [
    ('Low', 'High', 'Very Good'),
    ('Low', 'Medium', 'Good'),
    ('Low', 'Low', 'Poor'),
    ('Medium', 'High', 'Good'),
    ('Medium', 'Medium', 'Average'),
    ('Medium', 'Low', 'Poor'),
    ('High', 'High', 'Average'),
    ('High', 'Medium', 'Poor'),
    ('High', 'Low', 'Very Poor')
]

output_values = {
    'Very Good': 0.9,
    'Good': 0.75,
    'Average': 0.5,
    'Poor': 0.25,
    'Very Poor': 0.1
}

# Apply fuzzy inference to the best solution
cost_best = best_position[0] * Cost_range[1]  # Example cost based on best position
utilization_best = sum(best_position)  # Example utilization based on best position

rule_activations = fuzzy_inference(cost_best, utilization_best)

# Defuzzification process
def defuzzification(rule_activations):
    numerator = sum(activation * output_values[label] for activation, label in rule_activations)
    denominator = sum(activation for activation, _ in rule_activations)
    return numerator / denominator if denominator != 0 else 0

# Get the defuzzified result (final evaluation)
defuzzified_result = defuzzification(rule_activations)
print("Fuzzy Inference Results:")
print("Rule Activations:", rule_activations)
print("Defuzzified Output:", defuzzified_result)
```
# Step 17: Visualizing the Fuzzy Logic Results
```
x_cost = np.linspace(0.012, 0.1046, 100)
x_util = np.linspace(0.2, 1.0, 100)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(x_cost, [cost_low(x) for x in x_cost], label='Low')
plt.plot(x_cost, [cost_medium(x) for x in x_cost], label='Medium')
plt.plot(x_cost, [cost_high(x) for x in x_cost], label='High')
plt.title('Cost Membership Functions')
plt.xlabel('Cost')
plt.ylabel('Membership Degree')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(x_util, [util_low(x) for x in x_util], label='Low')
plt.plot(x_util, [util_medium(x) for x in x_util], label='Medium')
plt.plot(x_util, [util_high(x) for x in x_util], label='High')
plt.title('Utilization Membership Functions')
plt.xlabel('Utilization')
plt.ylabel('Membership Degree')
plt.legend()
plt.grid(True)
plt.show()
```
# Step 18: Comparison of PSO vs MOPSO
```
# Assuming pso_costs and pso_utils are defined from previous PSO analysis
import matplotlib.pyplot as plt

# Step 18: Comparison of PSO vs MOPSO
# Assuming pso_costs and pso_utils are defined from previous PSO analysis

pso_costs = [10, 15, 20, 25, 30]  # Example PSO costs
pso_utils = [0.5, 0.6, 0.7, 0.8, 0.9]  # Example PSO utilities

mopso_costs = [12, 18, 22, 28, 32]  # Example MOPSO costs
mopso_utils = [0.4, 0.5, 0.6, 0.7, 0.8]  # Example MOPSO utilities

# Plotting the comparison
plt.figure(figsize=(10, 6))
plt.scatter(pso_costs, pso_utils, color='blue', label='PSO', alpha=0.6)
plt.scatter(mopso_costs, mopso_utils, color='red', label='MOPSO', alpha=0.8)
plt.title('Comparison of PSO and MOPSO: Cost vs. Resource Utilization')
plt.xlabel('Cost')
plt.ylabel('Resource Utilization')
plt.legend()
plt.grid(True)
plt.show()
# plt.figure(figsize=(10, 6))
# plt.scatter(pso_costs, pso_utils, color='blue', label='PSO', alpha=0.6)
# plt.scatter(mopso_costs, mopso_utils, color='red', label='MOPSO', alpha=0.8)
# plt.title('Comparison of PSO and MOPSO: Cost vs. Resource Utilization')
# plt.xlabel('Cost')
# plt.ylabel('Resource Utilization')
# plt.legend()
# plt.grid(True)
# plt.show()
```
# Step 19: Final Comparison of PSO and MOPSO with Fuzzy Output
```
print(f"Final Fuzzy Evaluation for Best PSO Solution: {defuzzified_result}")

```
# Step 20: Saving Results to CSV for Future Analysis
```
import pandas as pd

# Create a dataframe to store the results
results_df = pd.DataFrame({
    'PSO Cost': pso_costs,
    'PSO Utilization': pso_utils,
    'MOPSO Cost': mopso_costs,
    'MOPSO Utilization': mopso_utils,
    'Defuzzified Result': [defuzzified_result] * len(pso_costs)  # Assuming same length for comparison
})

# Save the dataframe to a CSV file for future analysis
results_df.to_csv('optimization_results.csv', index=False)
print("Optimization results have been saved to 'optimization_results.csv'.")
```
# Step 21: Statistical Analysis of Results (Optional)
```
# Calculate some basic statistics for PSO and MOPSO performance
pso_cost_mean = np.mean(pso_costs)
pso_util_mean = np.mean(pso_utils)
mopso_cost_mean = np.mean(mopso_costs)
mopso_util_mean = np.mean(mopso_utils)

pso_cost_std = np.std(pso_costs)
pso_util_std = np.std(pso_utils)
mopso_cost_std = np.std(mopso_costs)
mopso_util_std = np.std(mopso_utils)

# Print out the statistics
print("\nPSO Mean and Std Deviation:")
print(f"Mean Cost: {pso_cost_mean}, Std: {pso_cost_std}")
print(f"Mean Utilization: {pso_util_mean}, Std: {pso_util_std}")

print("\nMOPSO Mean and Std Deviation:")
print(f"Mean Cost: {mopso_cost_mean}, Std: {mopso_cost_std}")
print(f"Mean Utilization: {mopso_util_mean}, Std: {mopso_util_std}")
```
# Step 22: Optimizing Further (Optional)
```
# Use the fuzzy inference results to refine the optimization parameters (e.g., particle velocity, position update)
# This is an advanced step and might require tuning the fuzzy logic or the PSO/MOPSO parameters.

# Example of refining parameters using defuzzified result
if defuzzified_result > 0.75:
    print("\nDefuzzified result is high. Consider reducing the exploration rate of PSO.")
    # Adjust PSO or MOPSO parameters based on fuzzy output (e.g., reduce exploration)
    # This part would depend on your specific problem and should be further refined for your application.
```
# Step 23: Conclusion
```
print("\nOptimization process has been completed.")
print(f"Best Solution Found (PSO): Position: {best_position}, Cost: {cost_best}, Utilization: {utilization_best}")
print(f"Defuzzified Evaluation: {defuzzified_result}")

# Final Output: This would typically be written to a log or report for documentation purposes
final_report = f"""
Optimization Process Completed

Best PSO Solution:
Position: {best_position}
Cost: {cost_best}
Utilization: {utilization_best}

Defuzzified Evaluation: {defuzzified_result}
"""

# Save the final report to a text file
with open("optimization_report.txt", "w") as file:
    file.write(final_report)

print("Optimization report has been saved to 'optimization_report.txt'.")
 ```
</p>
