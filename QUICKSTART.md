# Quick Start Guide

## 🚀 Get Started in 3 Steps

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Simulation
```bash
python economic_simulation.py
```

### 3. View Results
The simulation will generate:
- `economic_simulation_results.png` - Comprehensive visualizations
- `transaction_history.csv` - Complete transaction data

## 📊 What You'll See

### Console Output
- Real-time progress for each simulation tick
- Transaction counts and statistics
- Final agent inventory states
- Example LLM prompt template

### Visualizations (PNG file)
1. **Price Discovery Curve** - Shows how market prices evolve over time
2. **Transaction Volume** - Number of trades per tick
3. **Price Distribution by Persona** - How different personalities affect pricing
4. **Average Price by Resource** - Comparison between Wheat and Gold

## 🎮 Try Different Scenarios

### Increase Agents
```python
python -c "from economic_simulation import run_simulation, analyze_and_plot; model, df = run_simulation(num_agents=20, num_ticks=30); analyze_and_plot(model, df)"
```

### Run Custom Analysis
```bash
python example_usage.py
```

### Compare Personas
```python
python -c "from example_usage import persona_comparison_example; persona_comparison_example()"
```

## 🔧 Configuration

### Adjust Simulation Parameters
Edit `economic_simulation.py`:
```python
# In main() function
NUM_AGENTS = 10  # Default: 10
NUM_TICKS = 20   # Default: 20
```

### Add New Resources
```python
# In TradingAgent class
RESOURCES = ["Wheat", "Gold", "Iron", "Wood"]
```

### Create Custom Personas
```python
# In TradingAgent class
PERSONAS = ["Aggressive", "Conservative", "Cooperative", "Opportunistic"]
```

## 🎓 Understanding the Output

### Agent Personas

| Persona | Behavior | Trade Success |
|---------|----------|---------------|
| **Aggressive** | Demands high utility gains, tough negotiator | Fewer trades, higher prices |
| **Conservative** | Accepts fair trades, risk-averse | Moderate trades, fair prices |
| **Cooperative** | Generous, facilitates market | Many trades, lower prices |

### Key Metrics

- **Average Price**: Overall market price level (typically 0.5-1.5)
- **Transaction Volume**: Number of successful trades
- **Price Discovery**: How prices stabilize over time
- **Persona Distribution**: Trading patterns by agent type

## 🐛 Troubleshooting

### No Transactions Occurring
- Increase `NUM_TICKS` (simulation runs longer)
- Increase `NUM_AGENTS` (more trading opportunities)
- Check agent initial inventories are reasonable

### Dependencies Issues
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Visualization Not Saving
- Check write permissions in current directory
- Ensure matplotlib backend is configured
- Try running with `MPLBACKEND=Agg python economic_simulation.py`

## 📚 Next Steps

1. **Integrate Real LLM**: Replace rule-based logic with OpenAI/LangChain API
2. **Add More Resources**: Expand from 2 to N resource types
3. **Implement Advanced Markets**: Add auctions, order books, futures
4. **Add Learning**: Implement reinforcement learning for agent strategies
5. **Scale Up**: Run simulations with 100+ agents

## 💡 Example Use Cases

### Research
- Study emergent market behavior
- Test economic theories
- Analyze negotiation strategies

### Education
- Teach agent-based modeling
- Demonstrate game theory concepts
- Show supply-demand dynamics

### Development
- Prototype trading algorithms
- Test pricing strategies
- Experiment with market mechanisms

## 📞 Need Help?

Check the comprehensive [README.md](README.md) for:
- Detailed architecture documentation
- API reference
- Customization guide
- Contributing guidelines

---

Happy Simulating! 🎉
