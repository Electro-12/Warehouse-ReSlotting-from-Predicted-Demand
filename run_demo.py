from src.integration import IntegrationSimulator

if __name__ == "__main__":
    print("Executing Final Project Demo...")
    sim = IntegrationSimulator()
    sim.get_demand_profiles()
    sim.generate_layouts()
    sim.simulate_picker_distance()
    print("\nDemo complete. Please check the 'plots' folder for the generated visual evaluations.")