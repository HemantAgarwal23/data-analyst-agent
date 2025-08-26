# 📊 Data Analyst Agent

An intelligent data analysis tool that allows users to upload CSV/Excel files and query their data using natural language. Built with Streamlit and powered by AI for seamless data exploration.

## 🌟 Features

### 🤖 Natural Language to SQL
- Convert plain English questions into SQL queries
- Multiple AI prompt strategies for better accuracy
- Intelligent fallback system with pattern-based query construction
- Support for complex queries including aggregations, filtering, and sorting

### 📈 Interactive Visualizations
- Automatic chart generation based on query results
- Manual chart creation with multiple chart types (Bar, Line, Histogram, Scatter)
- Smart visualization selection based on data types
- Responsive charts with proper formatting

### 🛡️ Robust Data Processing
- Support for CSV and Excel file formats
- Automatic column name sanitization
- Data type inference and validation
- Comprehensive error handling with user-friendly messages

### 🔍 Advanced Query Capabilities
- **Aggregation queries**: Average, Sum, Count, Max, Min
- **Top N queries**: "Show top 5 records by price"
- **Filtering**: "Show products where category is Electronics"
- **Statistical analysis**: Automatic data profiling and summary statistics

## 🚀 Demo

### Sample Queries You Can Ask:
```
"Show top 5 products by price"
"Average sales by category"
"Count total orders"
"Show customers where region is North"
"Maximum revenue"
"Sum of quantities"
```

### Example Output:
![Demo Screenshot](https://github.com/HemantAgarwal23/data-analyst-agent/blob/main/Screenshot%202025-08-27%20010713.png)

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/HemantAgarwal23/data-analyst-agent.git
   cd data-analyst-agent
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If it doesn't open automatically, navigate to the URL manually

## 📦 Dependencies

```txt
streamlit>=1.28.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
transformers>=4.30.0
torch>=2.0.0
openpyxl>=3.1.0
sqlite3
regex>=2023.6.3
```

## 📊 Sample Data

The `/sample_data/` folder contains example datasets to test the application


### Sample Data Structure:
```csv
id,name,category,price,quantity
1,Apple,Fruit,50,100
2,Banana,Fruit,20,150
3,Carrot,Vegetable,30,200
4,Tomato,Vegetable,25,120
5,Milk,Dairy,60,80
6,Cheese,Dairy,120,40
```

## 🎯 Usage Guide

### 1. Upload Your Data
- Click "Upload CSV or Excel" button
- Select your data file (supports .csv and .xlsx formats)
- The app will automatically preview your data and show column information

### 2. Ask Natural Language Questions
- Type your question in the text input field
- Examples:
  - `"What is the average price?"`
  - `"Show top 10 products by sales"`
  - `"Count customers by region"`
  - `"Display items where category is Electronics"`

### 3. View Results
- The app generates and displays the SQL query
- Results are shown in a table format
- Automatic visualizations are created when appropriate

### 4. Create Custom Visualizations
- Use the manual visualization section
- Select chart type and axes
- Generate custom plots for deeper analysis

## 🏗️ Architecture

### System Components:
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit     │────│   Text-to-SQL    │────│   SQLite        │
│   Frontend      │    │   AI Model       │    │   Database      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                    ┌──────────────────┐
                    │   Visualization  │
                    │   Engine         │
                    └──────────────────┘
```

### Key Technologies:
- **Frontend**: Streamlit for interactive web interface
- **AI/ML**: Hugging Face Transformers for natural language processing
- **Database**: SQLite for in-memory data processing
- **Visualization**: Matplotlib + Seaborn for chart generation
- **Data Processing**: Pandas for data manipulation

## 🤖 AI Model Details

The application uses the `Tigran555/text2sql` model from Hugging Face, which is specifically trained for converting natural language to SQL queries. The system implements:

- **Multiple prompt strategies** for improved accuracy
- **Query validation** to ensure generated SQL is safe and valid
- **Fallback pattern matching** when the AI model fails
- **Context-aware prompting** with table schema information

## 🛡️ Security Features

- **SQL Injection Protection**: Only SELECT queries are allowed
- **Input Validation**: Malicious queries are blocked
- **Safe Execution**: All SQL runs in isolated SQLite environment
- **Error Handling**: Comprehensive error catching and user-friendly messages

## 🎨 Customization

### Adding New Chart Types
```python
# In the visualization section, add new chart types:
if chart_type == "Your_New_Chart":
    your_plotting_function(data, ax)
```

### Extending Query Patterns
```python
# In construct_sql_manually function, add new patterns:
if 'your_pattern' in question_lower:
    return f"SELECT ... FROM data WHERE ..."
```

## 🚧 Known Limitations

- **Model Accuracy**: Text-to-SQL conversion may not be perfect for complex queries
- **File Size**: Large files (>100MB) may cause performance issues
- **Chart Limits**: Automatic visualizations work best with <100 data points
- **Language**: Currently optimized for English queries only

## 🔮 Future Enhancements

- [ ] Support for multiple database connections (PostgreSQL, MySQL)
- [ ] Advanced chart types (3D plots, interactive charts)
- [ ] Query history and bookmarking
- [ ] Export functionality (PDF, PowerPoint reports)
- [ ] Multi-language support
- [ ] Real-time data streaming capabilities
- [ ] Advanced statistical analysis features
- [ ] User authentication and data privacy controls

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup:
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Submit a pull request with a clear description

### Contribution Guidelines:
- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation as needed
- Ensure all existing tests pass

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face** for providing the text-to-SQL model
- **Streamlit** for the amazing web framework
- **The open-source community** for various libraries used




**Made with ❤️ by Hemant Agarwal**

*If this project helped you, please consider giving it a ⭐ on GitHub!*
