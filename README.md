# Data Modeling Platform

A Streamlit-based data modeling application running on Databricks.

## Features

- 📊 Data Exploration with interactive visualizations
- 🤖 Multiple ML model types (Linear Regression, Random Forest, XGBoost, Neural Networks)
- 📈 Real-time predictions and model performance metrics
- 💾 Model export and versioning
- 🔄 Automatic deployment to Databricks via GitHub Actions

## Project Structure

```
.
├── src/
│   └── app.py              # Main Streamlit application
├── .github/
│   └── workflows/
│       └── deploy.yml      # Auto-deployment to Databricks
├── requirements.txt         # Python dependencies
└── README.md
```

## Development Workflow

1. **Edit code** using GitHub Codespaces, github.dev, or web interface
2. **Commit and push** to main branch
3. **Automatic deployment** to Databricks via GitHub Actions
4. **Access app** in Databricks Repos

## Running Locally (Optional)

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

## Running in Databricks

1. Navigate to Repos in your Databricks workspace
2. Pull latest changes
3. Run from notebook:

```python
%pip install -r requirements.txt
!streamlit run /Workspace/Repos/your-email/your-repo/src/app.py
```

## Technology Stack

- **Frontend**: Streamlit
- **Visualization**: Plotly
- **ML Libraries**: scikit-learn, pandas, numpy
- **Platform**: Databricks
- **CI/CD**: GitHub Actions

## License

MIT
