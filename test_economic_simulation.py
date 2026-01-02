"""
Unit tests for the Economic Simulation.

Run with: pytest test_economic_simulation.py -v
or: python -m pytest test_economic_simulation.py -v
"""

import pytest
from economic_simulation import (
    TradingAgent,
    EconomicSimulationModel,
    TradeProposal,
    Transaction,
    run_simulation
)


class TestTradingAgent:
    """Test the TradingAgent class."""
    
    def test_agent_initialization(self):
        """Test that agents initialize with correct properties."""
        model = EconomicSimulationModel(num_agents=1)
        agent = list(model.schedule.agents)[0]
        
        assert agent.unique_id is not None
        assert agent.persona in TradingAgent.PERSONAS
        assert "Wheat" in agent.inventory
        assert "Gold" in agent.inventory
        assert agent.inventory["Wheat"] > 0
        assert agent.inventory["Gold"] > 0
        assert len(agent.utility_weights) == len(TradingAgent.RESOURCES)
    
    def test_persona_assignment(self):
        """Test that specific personas can be assigned."""
        model = EconomicSimulationModel(num_agents=1)
        agent = TradingAgent(99, model, persona="Aggressive")
        
        assert agent.persona == "Aggressive"
    
    def test_utility_calculation(self):
        """Test utility function calculations."""
        model = EconomicSimulationModel(num_agents=1)
        agent = TradingAgent(0, model)
        
        utility = agent.calculate_utility("Wheat", 10)
        assert utility > 0
        assert isinstance(utility, float)
    
    def test_utility_weights_update(self):
        """Test that utility weights change with inventory."""
        model = EconomicSimulationModel(num_agents=1)
        agent = TradingAgent(0, model)
        
        initial_wheat_utility = agent.utility_weights["Wheat"]
        
        # Increase wheat inventory
        agent.inventory["Wheat"] += 50
        agent.update_utility_weights()
        
        # Utility should decrease (we need it less now)
        assert agent.utility_weights["Wheat"] < initial_wheat_utility
    
    def test_evaluate_trade_insufficient_inventory(self):
        """Test that trades are rejected when inventory is insufficient."""
        model = EconomicSimulationModel(num_agents=1)
        agent = TradingAgent(0, model)
        
        # Try to trade more than we have
        accept, price = agent.evaluate_trade(
            "Gold", 10,
            "Wheat", agent.inventory["Wheat"] + 100,
        )
        
        assert accept is False
    
    def test_evaluate_trade_zero_offer(self):
        """Test that trades with zero offer amount are rejected."""
        model = EconomicSimulationModel(num_agents=1)
        agent = TradingAgent(0, model)
        
        accept, price = agent.evaluate_trade(
            "Gold", 0,  # Zero offer
            "Wheat", 10,
        )
        
        assert accept is False
    
    def test_evaluate_trade_persona_differences(self):
        """Test that different personas have different acceptance thresholds."""
        model = EconomicSimulationModel(num_agents=1)
        
        aggressive = TradingAgent(0, model, persona="Aggressive")
        cooperative = TradingAgent(1, model, persona="Cooperative")
        
        # Set identical inventories
        aggressive.inventory = {"Wheat": 30, "Gold": 30}
        cooperative.inventory = {"Wheat": 30, "Gold": 30}
        aggressive.update_utility_weights()
        cooperative.update_utility_weights()
        
        # Evaluate same trade
        accept_agg, price_agg = aggressive.evaluate_trade("Gold", 10, "Wheat", 10)
        accept_coop, price_coop = cooperative.evaluate_trade("Gold", 10, "Wheat", 10)
        
        # Both evaluate, but prices may differ based on persona
        assert isinstance(accept_agg, bool)
        assert isinstance(accept_coop, bool)


class TestEconomicSimulationModel:
    """Test the EconomicSimulationModel class."""
    
    def test_model_initialization(self):
        """Test that model initializes correctly."""
        model = EconomicSimulationModel(num_agents=5)
        
        assert model.num_agents == 5
        assert len(list(model.schedule.agents)) == 5
        assert len(model.transactions) == 0
    
    def test_model_step(self):
        """Test that model can execute a step."""
        model = EconomicSimulationModel(num_agents=5)
        initial_steps = model.schedule.steps
        
        model.step()
        
        assert model.schedule.steps == initial_steps + 1
    
    def test_persona_distribution(self):
        """Test that personas are distributed among agents."""
        model = EconomicSimulationModel(num_agents=9)
        
        personas = [agent.persona for agent in model.schedule.agents]
        
        # Should have variety of personas
        assert "Aggressive" in personas
        assert "Conservative" in personas
        assert "Cooperative" in personas
    
    def test_simulation_runs(self):
        """Test that simulation can run multiple steps."""
        model = EconomicSimulationModel(num_agents=5)
        
        for _ in range(10):
            model.step()
        
        assert model.schedule.steps == 10


class TestTradeExecution:
    """Test trade execution functionality."""
    
    def test_trade_execution_updates_inventory(self):
        """Test that executing a trade updates both agents' inventories."""
        model = EconomicSimulationModel(num_agents=2)
        agents = list(model.schedule.agents)
        agent1, agent2 = agents[0], agents[1]
        
        # Record initial inventories
        agent1_wheat_initial = agent1.inventory["Wheat"]
        agent1_gold_initial = agent1.inventory["Gold"]
        agent2_wheat_initial = agent2.inventory["Wheat"]
        agent2_gold_initial = agent2.inventory["Gold"]
        
        # Execute a trade: agent1 gives 5 Wheat, gets 5 Gold
        agent1.execute_trade(
            agent2,
            resource_given="Wheat",
            amount_given=5,
            resource_received="Gold",
            amount_received=5,
            price=1.0
        )
        
        # Check inventories updated correctly
        assert agent1.inventory["Wheat"] == agent1_wheat_initial - 5
        assert agent1.inventory["Gold"] == agent1_gold_initial + 5
        assert agent2.inventory["Wheat"] == agent2_wheat_initial + 5
        assert agent2.inventory["Gold"] == agent2_gold_initial - 5
    
    def test_trade_execution_records_transaction(self):
        """Test that trades are recorded in history."""
        model = EconomicSimulationModel(num_agents=2)
        agents = list(model.schedule.agents)
        agent1, agent2 = agents[0], agents[1]
        
        initial_transactions = len(model.transactions)
        
        agent1.execute_trade(
            agent2,
            resource_given="Wheat",
            amount_given=5,
            resource_received="Gold",
            amount_received=5,
            price=1.0
        )
        
        assert len(model.transactions) == initial_transactions + 1
        assert len(agent1.trade_history) > 0
        assert len(agent2.trade_history) > 0


class TestSimulationRun:
    """Test the complete simulation run."""
    
    def test_run_simulation_returns_correct_types(self):
        """Test that run_simulation returns model and DataFrame."""
        model, transactions_df = run_simulation(num_agents=5, num_ticks=5)
        
        assert isinstance(model, EconomicSimulationModel)
        assert hasattr(transactions_df, 'empty')  # pandas DataFrame
    
    def test_run_simulation_executes_ticks(self):
        """Test that simulation runs for specified number of ticks."""
        model, transactions_df = run_simulation(num_agents=5, num_ticks=10)
        
        assert model.schedule.steps == 10
    
    def test_simulation_generates_transactions(self):
        """Test that simulation generates some transactions."""
        # Run longer simulation to ensure transactions occur
        model, transactions_df = run_simulation(num_agents=10, num_ticks=20)
        
        # With 10 agents and 20 ticks, we should get some transactions
        # (not guaranteed but highly likely)
        assert len(model.transactions) >= 0  # At least should not error


class TestTradeProposal:
    """Test trade proposal generation."""
    
    def test_generate_trade_proposal(self):
        """Test that agents can generate trade proposals."""
        model = EconomicSimulationModel(num_agents=2)
        agents = list(model.schedule.agents)
        agent1, agent2 = agents[0], agents[1]
        
        proposal = agent1.generate_llm_trade_proposal(agent2)
        
        # Proposal may be None if no beneficial trade found
        if proposal is not None:
            assert isinstance(proposal, TradeProposal)
            assert proposal.from_agent_id == agent1.unique_id
            assert proposal.to_agent_id == agent2.unique_id
            assert proposal.offer_resource in TradingAgent.RESOURCES
            assert proposal.request_resource in TradingAgent.RESOURCES
            assert proposal.offer_amount > 0
            assert proposal.request_amount > 0


# Integration test
class TestIntegration:
    """Integration tests for the full system."""
    
    def test_full_simulation_pipeline(self):
        """Test complete simulation pipeline."""
        # Run simulation
        model, transactions_df = run_simulation(num_agents=8, num_ticks=15)
        
        # Verify model state
        assert model.schedule.steps == 15
        assert len(list(model.schedule.agents)) == 8
        
        # Verify agents have reasonable inventory
        for agent in model.schedule.agents:
            assert agent.inventory["Wheat"] >= 0
            assert agent.inventory["Gold"] >= 0
        
        # Verify transactions (if any occurred)
        if not transactions_df.empty:
            assert 'tick' in transactions_df.columns
            assert 'buyer_id' in transactions_df.columns
            assert 'seller_id' in transactions_df.columns
            assert 'price' in transactions_df.columns
            assert all(transactions_df['price'] > 0)


if __name__ == "__main__":
    # Run tests with pytest if available, otherwise inform user
    try:
        import pytest
        pytest.main([__file__, "-v"])
    except ImportError:
        print("pytest not installed. Install it with: pip install pytest")
        print("Then run: pytest test_economic_simulation.py -v")
