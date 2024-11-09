from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

"""
BayesianNetwork: Used to define the structure of the Bayesian Network.
VariableElimination: An inference algorithm to perform queries on the network.
TabularCPD: Represents the Conditional Probability Distributions for each node.
"""

# u --> v, src to dest node
car_model = BayesianNetwork(
    [
        ("Battery", "Radio"),
        ("Battery", "Ignition"),
        ("Ignition","Starts"),
        ("Gas","Starts"),
        ("Starts","Moves"),
        ("KeyPresent", "Starts")  # New edge from KeyPresent to Starts
    ]
)

# Defining the parameters using CPT
from pgmpy.factors.discrete import TabularCPD

cpd_battery = TabularCPD(
    variable="Battery", variable_card=2, values=[[0.70], [0.30]],
    state_names={"Battery":['Works',"Doesn't work"]},
)

cpd_gas = TabularCPD(
    variable="Gas", variable_card=2, values=[[0.40], [0.60]],
    state_names={"Gas":['Full',"Empty"]},
)

cpd_radio = TabularCPD(
    variable=  "Radio", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Radio": ["turns on", "Doesn't turn on"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_ignition = TabularCPD(
    variable=  "Ignition", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Ignition": ["Works", "Doesn't work"],
                 "Battery": ['Works',"Doesn't work"]}
)

# Starts CPD updated to include KeyPresent
cpd_starts = TabularCPD(
    variable="Starts",
    variable_card=2,
    values=[
        # P(Starts='yes' | Ignition, Gas, KeyPresent)
        [0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        # P(Starts='no' | Ignition, Gas, KeyPresent)
        [0.01, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]
    ],
    evidence=["Ignition", "Gas", "KeyPresent"],
    evidence_card=[2, 2, 2],
    state_names={
        "Starts": ['yes', 'no'],
        "Ignition": ["Works", "Doesn't work"],
        "Gas": ['Full', "Empty"],
        "KeyPresent": ["yes", "no"]
    },
)

cpd_moves = TabularCPD(
    variable="Moves", variable_card=2,
    values=[[0.8, 0.01],[0.2, 0.99]],
    evidence=["Starts"],
    evidence_card=[2],
    state_names={"Moves": ["yes", "no"],
                 "Starts": ['yes', 'no'] }
)


# new CPD for KeyPresent
cpd_keypresent = TabularCPD(
    variable="KeyPresent",
    variable_card=2,
    values=[[0.7], [0.3]],
    state_names={"KeyPresent": ["yes", "no"]}
)




# Associating the parameters with the model structure, add cpd_keypresent
car_model.add_cpds( cpd_starts, cpd_ignition, cpd_gas, cpd_radio, cpd_battery, cpd_moves, cpd_keypresent)

car_infer = VariableElimination(car_model)

print(car_infer.query(variables=["Moves"],evidence={"Radio":"turns on", "Starts":"yes"}))

# 1. Probability that Battery is Not Working Given Car Will Not Move
result1 = car_infer.query(
    variables=["Battery"],
    evidence={"Moves": "no"}
)

# 2. Probability that Car Will Not Start Given Radio is Not Working
result2 = car_infer.query(
    variables=["Starts"],
    evidence={"Radio": "Doesn't turn on"}
)

# 3. Effect of Gas on Radio Working Given Battery is Working
result3 = car_infer.query(
    variables=["Radio"],
    evidence={"Battery": "Works", "Gas": "Full"}
)
result4 = car_infer.query(
    variables=["Radio"],
    evidence={"Battery": "Works", "Gas": "Empty"}
)

# 4. Probability of Ignition Failing Given Car Doesn't Move and No Gas
result5 = car_infer.query(
    variables=["Ignition"],
    evidence={"Moves": "no", "Gas": "Empty"}
)

# 5. Probability that Car Starts Given Radio Works and Has Gas
result6 = car_infer.query(
    variables=["Starts"],
    evidence={"Radio": "turns on", "Gas": "Full"}
)

# 6. Probability that the Key is Not Present Given that the Car Does Not Move
result7 = car_infer.query(
    variables=["KeyPresent"],
    evidence={"Moves": "no"}
)

# Main Function to Execute Queries
if __name__ == '__main__':
    print("----- Bayesian Network Queries -----\n")

    print("1. Probability that Battery is Not Working Given Car Will Not Move")
    print(result1)
    print("\n--------------------------------------\n")

    print("2. Probability that Car Will Not Start Given Radio is Not Working")
    print(result2)
    print("\n--------------------------------------\n")

    print("3. Effect of Gas on Radio Working Given Battery is Working")
    print("   a) Gas is Full:")
    print(result3)
    print("   b) Gas is Empty:")
    print(result4)
    print("\n--------------------------------------\n")

    print("4. Probability of Ignition Failing Given Car Doesn't Move and No Gas")
    print(result5)
    print("\n--------------------------------------\n")

    print("5. Probability that Car Starts Given Radio Works and Has Gas")
    print(result6)
    print("\n--------------------------------------\n")

    print("6. Probability that the Key is Not Present Given that the Car Does Not Move")
    print(result7)
    print("\n--------------------------------------\n")