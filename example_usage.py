"""
Example usage of the Economic Simulation API.

This script demonstrates how to use the simulation programmatically
and perform custom analysis on the results.
"""

from economic_simulation import (
    run_simulation,
    analyze_and_plot,
    EconomicSimulationModel,
    TradingAgent
)
import pandas as pd


def custom_analysis_example():
    """Run simulation with custom analysis."""
    print("Running Economic Simulation with Custom Analysis\n")
    
    # Run simulation with custom parameters
    model, transactions_df = run_simulation(num_agents=12, num_ticks=25)
    
    if transactions_df.empty:
        print("No transactions occurred. Try increasing num_ticks or num_agents.")
        return
    
    # Custom Analysis 1: Persona Performance
    print("\n" + "="*70)
    print("CUSTOM ANALYSIS: PERSONA PERFORMANCE")
    print("="*70)
    
    persona_stats = transactions_df.groupby('buyer_persona').agg({
        'price': ['mean', 'std', 'count'],
        'amount': 'sum'
    })
    print("\nBuyer Persona Statistics:")
    print(persona_stats)
    
    # Custom Analysis 2: Agent-specific trading patterns
    print("\n" + "="*70)
    print("CUSTOM ANALYSIS: TOP TRADERS")
    print("="*70)
    
    buyer_counts = transactions_df['buyer_id'].value_counts().head(5)
    print("\nMost Active Buyers:")
    for agent_id, count in buyer_counts.items():
        agent = next(a for a in model.schedule.agents if a.unique_id == agent_id)
        print(f"  Agent {agent_id} ({agent.persona}): {count} purchases")
    
    # Custom Analysis 3: Price evolution by resource
    print("\n" + "="*70)
    print("CUSTOM ANALYSIS: PRICE EVOLUTION")
    print("="*70)
    
    for resource in transactions_df['resource'].unique():
        resource_df = transactions_df[transactions_df['resource'] == resource]
        early_price = resource_df[resource_df['tick'] <= 5]['price'].mean()
        late_price = resource_df[resource_df['tick'] > 15]['price'].mean()
        
        print(f"\n{resource}:")
        print(f"  Early average (ticks 1-5): {early_price:.2f}")
        print(f"  Late average (ticks 16+): {late_price:.2f}")
        print(f"  Price change: {((late_price - early_price) / early_price * 100):.1f}%")
    
    # Generate standard visualizations
    analyze_and_plot(model, transactions_df)
    
    print("\n" + "="*70)
    print("Custom analysis complete!")
    print("="*70)


def persona_comparison_example():
    """Compare different persona distributions."""
    print("\n" + "="*70)
    print("PERSONA DISTRIBUTION COMPARISON")
    print("="*70)
    
    scenarios = [
        ("All Aggressive", ["Aggressive"] * 10),
        ("All Conservative", ["Conservative"] * 10),
        ("All Cooperative", ["Cooperative"] * 10),
        ("Mixed", ["Aggressive", "Conservative", "Cooperative"] * 3 + ["Aggressive"]),
    ]
    
    results = []
    
    for scenario_name, personas in scenarios:
        print(f"\nRunning scenario: {scenario_name}")
        
        # Create model with specific personas
        model = EconomicSimulationModel(num_agents=len(personas))
        
        # Override default personas
        for i, agent in enumerate(model.schedule.agents):
            agent.persona = personas[i]
        
        # Run simulation
        for _ in range(15):
            model.step()
        
        # Convert transactions to DataFrame for analysis
        if model.transactions:
            transactions_df = pd.DataFrame([
                {'price': t.price} for t in model.transactions
            ])
        else:
            transactions_df = pd.DataFrame()
        
        # Collect results
        avg_price = transactions_df['price'].mean() if len(model.transactions) > 0 else 0
        
        results.append({
            'scenario': scenario_name,
            'transactions': len(model.transactions),
            'avg_price': avg_price
        })
        
        print(f"  Transactions: {len(model.transactions)}")
        print(f"  Average Price: {avg_price:.2f}")
    
    # Summary
    print("\n" + "="*70)
    print("SCENARIO COMPARISON SUMMARY")
    print("="*70)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))


def main():
    """Run example analyses."""
    print("\n" + "="*70)
    print("ECONOMIC SIMULATION - EXAMPLE USAGE")
    print("="*70)
    
    # Example 1: Custom analysis
    custom_analysis_example()
    
    # Example 2: Persona comparison
    # Uncomment to run:
    # persona_comparison_example()


if __name__ == "__main__":
    main()
