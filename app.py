import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
import sqlite3
import re

# -----------------------------------------------------------
# Title + Sidebar
# -----------------------------------------------------------
st.set_page_config(page_title="Data Analyst Agent", layout="wide")
st.title("📊 Data Analyst Agent")

st.sidebar.header("ℹ️ Info")
st.sidebar.markdown("This app lets you **upload a CSV/Excel file** and ask questions in natural language. It will:")
st.sidebar.markdown("- Convert your question → SQL query")
st.sidebar.markdown("- Run SQL on your data")
st.sidebar.markdown("- Show results + charts 🎉")

# -----------------------------------------------------------
# Load HF model (no token needed)
# -----------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        return pipeline("text2text-generation", model="Tigran555/text2sql")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def construct_sql_manually(question, columns):
    """Manual SQL construction as fallback when model fails"""
    question_lower = question.lower().strip()
    
    # Common query patterns
    
    # 1. Top N queries
    if 'top' in question_lower:
        # Extract number
        import re
        numbers = re.findall(r'\d+', question)
        limit = numbers[0] if numbers else '10'
        
        # Find column to order by
        order_col = None
        for col in columns:
            if col.lower() in question_lower:
                order_col = col
                break
        
        if order_col:
            return f"SELECT * FROM data ORDER BY {order_col} DESC LIMIT {limit}"
        else:
            return f"SELECT * FROM data LIMIT {limit}"
    
    # 2. Average queries
    if 'average' in question_lower or 'avg' in question_lower:
        # Find numeric column
        for col in columns:
            if col.lower() in question_lower:
                return f"SELECT AVG({col}) as average_{col} FROM data"
        # If no specific column, try price if it exists
        if 'price' in [c.lower() for c in columns]:
            price_col = next(c for c in columns if c.lower() == 'price')
            return f"SELECT AVG({price_col}) as average_{price_col} FROM data"
    
    # 3. Count queries
    if 'count' in question_lower:
        return "SELECT COUNT(*) as total_count FROM data"
    
    # 4. Sum queries
    if 'sum' in question_lower or 'total' in question_lower:
        for col in columns:
            if col.lower() in question_lower:
                return f"SELECT SUM({col}) as total_{col} FROM data"
    
    # 5. Show/Select with WHERE
    if 'show' in question_lower and 'where' in question_lower:
        parts = question_lower.split('where')
        if len(parts) == 2:
            select_part = parts[0].replace('show', '').strip()
            where_part = parts[1].strip()
            
            # Find column names in select part
            select_cols = []
            for col in columns:
                if col.lower() in select_part.lower():
                    select_cols.append(col)
            
            if not select_cols:
                select_cols = ['*']
            
            # Parse WHERE condition
            if ' is ' in where_part:
                condition_parts = where_part.split(' is ')
                if len(condition_parts) == 2:
                    where_col = condition_parts[0].strip()
                    where_val = condition_parts[1].strip()
                    
                    # Find matching column
                    actual_col = None
                    for col in columns:
                        if col.lower() == where_col.lower():
                            actual_col = col
                            break
                    
                    if actual_col:
                        # Check if value should be quoted
                        if where_val.replace('.', '').replace('-', '').isdigit():
                            where_clause = f"{actual_col} = {where_val}"
                        else:
                            where_clause = f"{actual_col} = '{where_val}'"
                        
                        select_clause = ', '.join(select_cols) if select_cols != ['*'] else '*'
                        return f"SELECT {select_clause} FROM data WHERE {where_clause}"
    
    # 6. Simple show/select all
    if 'show' in question_lower or 'select' in question_lower:
        if 'all' in question_lower:
            return "SELECT * FROM data"
        
        # Look for specific columns
        select_cols = []
        for col in columns:
            if col.lower() in question_lower:
                select_cols.append(col)
        
        if select_cols:
            return f"SELECT {', '.join(select_cols)} FROM data"
    
    # 7. Max/Min queries
    if 'maximum' in question_lower or 'max' in question_lower:
        for col in columns:
            if col.lower() in question_lower:
                return f"SELECT MAX({col}) as max_{col} FROM data"
    
    if 'minimum' in question_lower or 'min' in question_lower:
        for col in columns:
            if col.lower() in question_lower:
                return f"SELECT MIN({col}) as min_{col} FROM data"
    
    return None

def clean_sql_query(sql_query):
    """Enhanced SQL query cleaning"""
    if not sql_query:
        return None
        
    # Remove any extra text that might be added by the model
    sql_query = sql_query.strip()
    
    # Remove common prefixes the model might add
    prefixes_to_remove = [
        "sql query:", "sql:", "query:", "answer:", "result:", 
        "the sql query is:", "here is the sql:", "sql code:"
    ]
    
    for prefix in prefixes_to_remove:
        if sql_query.lower().startswith(prefix):
            sql_query = sql_query[len(prefix):].strip()
    
    # Extract SQL from various formats
    sql_patterns = [
        r'SELECT.*?(?:;|$)',  # Standard SELECT
        r'select.*?(?:;|$)',  # Lowercase SELECT
    ]
    
    for pattern in sql_patterns:
        match = re.search(pattern, sql_query, re.IGNORECASE | re.DOTALL)
        if match:
            sql_query = match.group(0)
            break
    
    # Clean up
    sql_query = sql_query.rstrip(';').strip()
    
    # Basic validation
    if not sql_query.upper().strip().startswith('SELECT'):
        return None
    
    return sql_query

def execute_sql_safely(sql_query, conn):
    """Execute SQL with better error handling"""
    try:
        # Additional safety check
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE']
        sql_upper = sql_query.upper()
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                return None, f"Dangerous SQL operation detected: {keyword}"
        
        result = pd.read_sql_query(sql_query, conn)
        return result, None
    except Exception as e:
        return None, str(e)

nlp_to_sql = load_model()
if nlp_to_sql:
    st.sidebar.success("✅ Model loaded successfully!")
else:
    st.sidebar.error("❌ Failed to load model")

# -----------------------------------------------------------
# File uploader
# -----------------------------------------------------------
st.subheader("📤 Upload your data")
uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

df = None
if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Check if data was parsed correctly
        if len(df.columns) == 1 and len(df.columns[0]) > 20:
            st.warning("⚠️ It looks like your CSV wasn't parsed correctly. The data might not have proper delimiters.")
            st.write("**Detected issue:** All data appears to be in one column.")
            st.write("**Original column name:**", df.columns[0])
            
            # Try to auto-fix common issues
            if st.button("🔧 Try to auto-fix CSV parsing"):
                raw_data = uploaded_file.getvalue().decode('utf-8')
                st.text_area("Raw file content:", raw_data[:500] + "..." if len(raw_data) > 500 else raw_data, height=100)
                
                # If it's the sample data format you showed, try to parse it manually
                lines = raw_data.strip().split('\n')
                if len(lines) > 0:
                    # Try different parsing strategies
                    parsed_data = try_manual_parsing(raw_data)
                    if parsed_data is not None:
                        df = parsed_data
                        st.success("✅ Successfully parsed the data!")
        
        st.success(f"✅ Loaded `{uploaded_file.name}` successfully!")
        
        # Clean column names (remove special characters, spaces)
        df.columns = df.columns.str.replace(' ', '_')
        df.columns = df.columns.str.replace('[^a-zA-Z0-9_]', '', regex=True)
        
    except Exception as e:
        st.error(f"Error reading file: {e}")

# -----------------------------------------------------------
# If file is uploaded → show schema + analysis
# -----------------------------------------------------------
if df is not None:
    st.write("### 🔍 Data Preview")
    st.dataframe(df.head())

    st.write("### 📑 Schema Inference")
    schema_info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Null Values": df.isnull().sum()
    })
    st.dataframe(schema_info)

    # Basic stats
    st.write("### 📊 Statistical Summary")
    st.write(df.describe(include="all"))

    # -------------------------------------------------------
    # Natural Language Query → SQL
    # -------------------------------------------------------
    st.subheader("💬 Ask Questions in Natural Language")
    
    # Show example questions
    st.write("**Example questions you can ask:**")
    st.write("- Show me all records")
    st.write("- Top 5 price")
    st.write("- Average price")
    st.write("- Count rows")
    st.write("- Show name where category is Fruit")
    st.write("- Maximum price")
    st.write("- Sum of quantity")
    
    question = st.text_input("Type your question about the data:")

    if question and nlp_to_sql:
        # Build schema string for model with proper column names
        schema_str = ", ".join([f"{col} ({dtype})" for col, dtype in zip(df.columns, df.dtypes)])
        
        # Multiple prompt strategies for better results
        prompt_v1 = f"""Generate SQL query for this question about table 'data':
Question: {question}
Available columns: {', '.join(df.columns)}
Table name: data
Only return the SQL SELECT statement:"""

        prompt_v2 = f"""# SQL Query Generator
Table: data
Schema: {schema_str}
Question: {question}
SQL Query (SELECT only):"""

        prompt_v3 = f"""Convert to SQL:
"{question}"
Table: data with columns: {', '.join(df.columns)}
Answer:"""

        try:
            # Try multiple prompt strategies
            cleaned_sql = None
            prompts_to_try = [prompt_v1, prompt_v2, prompt_v3]
            
            for i, prompt in enumerate(prompts_to_try):
                try:
                    with st.spinner(f"Generating SQL query (attempt {i+1}/3)..."):
                        result = nlp_to_sql(prompt, max_length=80, do_sample=False, temperature=0.1)
                        sql_query = result[0]["generated_text"]
                    
                    # Show raw output for debugging
                    if i == 0:  # Only show for first attempt
                        with st.expander("🔍 Debug: Raw model output"):
                            st.code(sql_query)
                    
                    # Clean the generated SQL
                    cleaned_sql = clean_sql_query(sql_query)
                    
                    # Validate that it uses our table and columns
                    if cleaned_sql:
                        # Check if query mentions our actual table and columns
                        sql_lower = cleaned_sql.lower()
                        uses_correct_table = 'data' in sql_lower
                        uses_actual_columns = any(col.lower() in sql_lower for col in df.columns)
                        
                        if uses_correct_table and uses_actual_columns:
                            st.success(f"✅ Generated valid SQL on attempt {i+1}")
                            break
                        else:
                            cleaned_sql = None
                            
                except Exception as e:
                    st.warning(f"Attempt {i+1} failed: {e}")
                    continue
            
            # If all attempts failed, try manual SQL construction
            if cleaned_sql is None:
                st.warning("🔧 Model failed to generate proper SQL. Trying pattern-based construction...")
                cleaned_sql = construct_sql_manually(question, df.columns)
                if cleaned_sql:
                    st.info("✅ Used pattern-based SQL generation")
            
            if cleaned_sql is None:
                st.error("⚠️ Could not generate a valid SQL query. Please try rephrasing your question.")
            else:
                st.write("### 📝 Generated SQL Query")
                st.code(cleaned_sql, language="sql")

                # Execute SQL on in-memory SQLite
                conn = sqlite3.connect(":memory:")
                df.to_sql("data", conn, index=False, if_exists="replace")
                
                result_df, error = execute_sql_safely(cleaned_sql, conn)
                
                if error:
                    st.error(f"⚠️ SQL execution error: {error}")
                    st.write("**Possible solutions:**")
                    st.write("- Check if column names in your question match the actual column names")
                    st.write("- Try rephrasing your question")
                    st.write("- Use simpler language")
                else:
                    st.write("### ✅ Query Result")
                    st.dataframe(result_df)

                    # Auto chart if numeric and reasonable size
                    if result_df.shape[1] == 2 and len(result_df) <= 100:
                        try:
                            # Check if we can create a meaningful chart
                            x_col, y_col = result_df.columns
                            if pd.api.types.is_numeric_dtype(result_df[y_col]):
                                fig, ax = plt.subplots(figsize=(10, 6))
                                
                                # Choose appropriate chart based on data
                                if len(result_df) <= 20:
                                    sns.barplot(data=result_df, x=x_col, y=y_col, ax=ax)
                                    plt.xticks(rotation=45)
                                else:
                                    sns.lineplot(data=result_df, x=x_col, y=y_col, ax=ax)
                                
                                plt.title(f"{y_col} by {x_col}")
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close()
                        except Exception as chart_error:
                            st.write(f"Could not create automatic chart: {chart_error}")
                
                conn.close()
                
        except Exception as e:
            st.error(f"⚠️ Failed to generate SQL: {e}")
            st.write("**Troubleshooting tips:**")
            st.write("- Make sure your question is clear and specific")
            st.write("- Reference actual column names from your data")
            st.write("- Try simpler questions first")

    # -------------------------------------------------------
    # Manual visualization options
    # -------------------------------------------------------
    st.subheader("📈 Create Your Own Visualization")

    chart_type = st.selectbox("Select chart type", ["Bar", "Line", "Histogram", "Scatter"])
    
    if chart_type in ["Bar", "Line", "Scatter"]:
        x_axis = st.selectbox("X-axis", options=df.columns)
        y_axis = st.selectbox("Y-axis", options=df.columns)
    else:  # Histogram
        x_axis = st.selectbox("Column for histogram", options=df.columns)
        y_axis = None

    if st.button("Generate Chart"):
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if chart_type == "Bar":
                # Limit to top 20 categories to avoid overcrowding
                if df[x_axis].nunique() > 20:
                    top_categories = df[x_axis].value_counts().head(20).index
                    plot_data = df[df[x_axis].isin(top_categories)]
                else:
                    plot_data = df
                sns.barplot(data=plot_data, x=x_axis, y=y_axis, ax=ax)
                plt.xticks(rotation=45)
                
            elif chart_type == "Line":
                sns.lineplot(data=df, x=x_axis, y=y_axis, ax=ax)
                
            elif chart_type == "Histogram":
                sns.histplot(df[x_axis], ax=ax, bins=30)
                
            elif chart_type == "Scatter":
                sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)
            
            plt.title(f"{chart_type} Chart: {x_axis}" + (f" vs {y_axis}" if y_axis else ""))
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        except Exception as e:
            st.error(f"Error creating chart: {e}")

else:
    st.info("👆 Please upload a CSV or Excel file to get started!")

# -----------------------------------------------------------
# Additional Tips Section
# -----------------------------------------------------------
if df is not None:
    with st.expander("💡 Tips for Better Results"):
        st.write("**For better SQL generation:**")
        st.write("- Use column names exactly as shown in the schema")
        st.write("- Start with simple questions like 'show all data' or 'count rows'")
        st.write("- Be specific about what you want to see")
        st.write("- Avoid complex joins or subqueries")
        st.write("")
        st.write("**For better results, try these exact phrases:**")
        for col in df.columns[:3]:  # Show examples for first 3 columns
            st.write(f"- Average {col}")
            st.write(f"- Top 5 {col}")
            st.write(f"- Show {col} where [other_column] is [value]")
        st.write("- Show all records")
        st.write("- Count rows")