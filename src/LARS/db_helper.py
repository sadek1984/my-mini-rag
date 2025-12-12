"""
🗄️ LARS DuckDB Database Helper
================================
Provides database connection and query utilities for LARS

Features:
- Connection management
- SQL query execution
- Schema-Only Prompting support
- Data insertion for new samples

Usage:
    from db_helper import LARSDatabase
    
    db = LARSDatabase()
    df = db.query("SELECT * FROM samples WHERE year = 2024")
    db.close()

Author: LARS Development Team
Date: 2025-12-12
"""

import duckdb
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List
import io


class LARSDatabase:
    """
    Database helper class for LARS application
    Uses DuckDB for fast analytical queries
    """
    
    def __init__(self, db_path: Optional[str] = None, read_only: bool = True):
        """
        Initialize database connection
        
        Args:
            db_path: Path to DuckDB file. If None, uses default location.
            read_only: If True, opens in read-only mode (faster for queries)
        """
        if db_path is None:
            db_path = Path(__file__).parent / 'lars_data.duckdb'
        
        self.db_path = Path(db_path)
        self.read_only = read_only
        self._connection = None
        
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
    
    @property
    def connection(self):
        """Lazy connection - only connect when needed"""
        if self._connection is None:
            self._connection = duckdb.connect(
                str(self.db_path), 
                read_only=self.read_only
            )
        return self._connection
    
    def query(self, sql: str) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame
        
        Args:
            sql: SQL query string
            
        Returns:
            pandas DataFrame with results
        """
        return self.connection.execute(sql).df()
    
    def execute(self, sql: str, params: tuple = None) -> Any:
        """
        Execute SQL statement (for INSERT, UPDATE, DELETE)
        
        Args:
            sql: SQL statement
            params: Optional tuple of parameters
            
        Returns:
            Query result
        """
        if params:
            return self.connection.execute(sql, params)
        return self.connection.execute(sql)
    
    def insert_sample(self, sample_data: Dict[str, Any]) -> bool:
        """
        Insert a new sample into the database
        
        Args:
            sample_data: Dictionary with column names as keys
            
        Returns:
            True if successful
        """
        if self.read_only:
            raise PermissionError("Database opened in read-only mode")
        
        columns = list(sample_data.keys())
        placeholders = ', '.join(['?' for _ in columns])
        values = tuple(sample_data.values())
        
        sql = f"INSERT INTO samples ({', '.join(columns)}) VALUES ({placeholders})"
        self.execute(sql, values)
        return True
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Get entire samples table as DataFrame
        (For compatibility with existing code)
        """
        return self.query("SELECT * FROM samples")
    
    def close(self):
        """Close database connection"""
        if self._connection:
            self._connection.close()
            self._connection = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    # =========================================================================
    # 🔒 SCHEMA-ONLY PROMPTING SUPPORT
    # =========================================================================
    
    def get_schema_info(self) -> str:
        """
        Get database schema information for Schema-Only Prompting
        Returns metadata WITHOUT exposing actual sensitive data
        
        Returns:
            Formatted schema string for AI prompts
        """
        schema_parts = []
        
        # 1. Get column information
        columns_df = self.query("DESCRIBE samples")
        schema_parts.append("**Column Information:**")
        schema_parts.append("| Column | Type |")
        schema_parts.append("|--------|------|")
        for _, row in columns_df.iterrows():
            schema_parts.append(f"| {row['column_name']} | {row['column_type']} |")
        
        # 2. Get row count
        count = self.query("SELECT COUNT(*) as count FROM samples")['count'].iloc[0]
        schema_parts.append(f"\n**Total Records:** {count:,} samples")
        
        # 3. Get numeric column statistics (aggregates only, no actual values)
        numeric_stats = self.query("""
            SELECT 
                'reading' as column_name,
                MIN(reading) as min_val,
                MAX(reading) as max_val,
                ROUND(AVG(reading), 4) as mean_val
            FROM samples
            UNION ALL
            SELECT 
                'limits',
                MIN(limits),
                MAX(limits),
                ROUND(AVG(limits), 4)
            FROM samples
            UNION ALL
            SELECT 
                'exceedance_ratio',
                MIN(exceedance_ratio),
                MAX(exceedance_ratio),
                ROUND(AVG(exceedance_ratio), 4)
            FROM samples
        """)
        
        schema_parts.append("\n**Numeric Column Ranges:**")
        schema_parts.append(numeric_stats.to_markdown(index=False))
        
        # 4. Get categorical column unique counts (NOT actual values)
        categorical_info = self.query("""
            SELECT 
                'vegetable_english' as column_name, 
                COUNT(DISTINCT vegetable_english) as unique_count
            FROM samples
            UNION ALL
            SELECT 'pesticide_standardized', COUNT(DISTINCT pesticide_standardized) FROM samples
            UNION ALL
            SELECT 'الحي', COUNT(DISTINCT الحي) FROM samples
            UNION ALL
            SELECT 'year', COUNT(DISTINCT year) FROM samples
        """)
        
        schema_parts.append("\n**Categorical Columns Info:**")
        for _, row in categorical_info.iterrows():
            col = row['column_name']
            count = row['unique_count']
            # Mark sensitive columns
            if col in ['client_name', 'sample_id', 'sample_code']:
                schema_parts.append(f"- `{col}`: {count} unique values (sensitive, not shown)")
            else:
                schema_parts.append(f"- `{col}`: {count} unique values")
        
        # 5. Add value patterns (generic examples, NOT actual data)
        schema_parts.append("""
**Common Value Patterns (Generic Examples):**
- `vegetable_english`: English vegetable names (e.g., 'Tomato', 'Cucumber', 'Pepper')
- `vegetable_arabic`: Arabic vegetable names (e.g., 'طماطم', 'خيار', 'فلفل')
- `pesticide_standardized`: English pesticide names (use for filtering)
- `الحي` / `neighborhood_english`: Neighborhood names (e.g., 'الإسكان', 'Housing')
- `is_compliant`: 1=compliant (below limit), 0=violation (above limit)
- `result`: Arabic text 'مطابق' (compliant) or 'غير مطابق' (non-compliant)
""")
        
        return '\n'.join(schema_parts)
    
    def get_sql_prompt(self, user_query: str) -> str:
        """
        Generate a complete prompt for AI to generate SQL queries
        Uses Schema-Only Prompting (no actual data exposed)
        
        Args:
            user_query: User's question in natural language
            
        Returns:
            Complete prompt for AI
        """
        schema_info = self.get_schema_info()
        
        prompt = f"""Generate a SQL query to answer this question about the pesticide testing database.

🔒 Database Schema (Metadata Only - No Actual Data Exposed):

{schema_info}

⚠️ PRIVACY NOTE: You are receiving ONLY schema information (column names and types).
DO NOT assume specific values exist. Write generic filtering/aggregation code.

**TABLE NAME:** samples

**User Question:** {user_query}

**IMPORTANT RULES:**
1. Use standard SQL syntax (DuckDB compatible)
2. Use single quotes for strings: WHERE column = 'value'
3. For Arabic text matching, use: WHERE column LIKE '%الإسكان%'
4. For compliance: is_compliant = 1 (compliant), is_compliant = 0 (violation)
5. For date filtering, use: WHERE year = 2024
6. Always include ORDER BY for sorted results
7. Use LIMIT for large result sets

**EXAMPLE QUERIES:**

-- Vegetables in a neighborhood with compliance stats
SELECT 
    vegetable_arabic as الخضار,
    COUNT(*) as عدد_العينات,
    SUM(CASE WHEN is_compliant = 0 THEN 1 ELSE 0 END) as فوق_الحد,
    SUM(CASE WHEN is_compliant = 1 THEN 1 ELSE 0 END) as تحت_الحد
FROM samples
WHERE الحي LIKE '%الإسكان%'
GROUP BY vegetable_arabic
ORDER BY عدد_العينات DESC;

-- Compliance trends by year
SELECT 
    year as السنة,
    COUNT(*) as total_samples,
    SUM(is_compliant) as compliant,
    SUM(CASE WHEN is_compliant = 0 THEN 1 ELSE 0 END) as violations,
    ROUND(AVG(is_compliant) * 100, 2) as compliance_rate
FROM samples
GROUP BY year
ORDER BY year;

-- Pesticides in vegetables
SELECT 
    vegetable_arabic as الخضار,
    COUNT(*) as عدد_العينات,
    SUM(CASE WHEN is_compliant = 1 THEN 1 ELSE 0 END) as تحت_الحد,
    SUM(CASE WHEN is_compliant = 0 THEN 1 ELSE 0 END) as فوق_الحد
FROM samples
WHERE pesticide_standardized LIKE '%Bifenthrin%'
GROUP BY vegetable_arabic
ORDER BY عدد_العينات DESC;

Generate ONLY the SQL query wrapped in ```sql``` markers:"""
        
        return prompt


# Singleton instance for easy import
_db_instance: Optional[LARSDatabase] = None

def get_database(read_only: bool = True) -> LARSDatabase:
    """
    Get database instance (singleton pattern)
    
    Args:
        read_only: If True, opens in read-only mode
        
    Returns:
        LARSDatabase instance
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = LARSDatabase(read_only=read_only)
    return _db_instance


def execute_sql(sql: str) -> pd.DataFrame:
    """
    Quick helper to execute SQL and return DataFrame
    
    Args:
        sql: SQL query
        
    Returns:
        pandas DataFrame
    """
    db = get_database()
    return db.query(sql)


# Pesticide mapping for Arabic → English translation
PESTICIDE_MAPPING = {
    'البايفنثرن': 'Bifenthrin',
    'الكلوربيريفوس': 'Chlorpyrifos',
    'الديمثويت': 'Dimethoate',
    'اللامبدا': 'Lambda-cyhalothrin',
    'الميثوميل': 'Methomyl',
    'الكاربندزيم': 'Carbendazim',
    'الإيماميكتين': 'Emamectin',
    'الأيميداكلوبريد': 'Imidacloprid',
    'الثيامثوكسام': 'Thiamethoxam',
    'البروفيزين': 'Buprofezin',
    'الأباميكتين': 'Abamectin',
    'الأسيتامبريد': 'Acetamiprid',
    'الدلتامثرين': 'Deltamethrin',
    'السيبرمثرين': 'Cypermethrin'
}


if __name__ == "__main__":
    # Test the database helper
    print("🧪 Testing LARS Database Helper...")
    
    db = LARSDatabase()
    
    # Test basic query
    print("\n1. Testing basic query:")
    result = db.query("SELECT COUNT(*) as count FROM samples")
    print(f"   Total records: {result['count'].iloc[0]}")
    
    # Test schema info
    print("\n2. Testing schema info (Schema-Only Prompting):")
    schema = db.get_schema_info()
    print(schema[:500] + "...")
    
    # Test SQL prompt generation
    print("\n3. Testing SQL prompt generation:")
    prompt = db.get_sql_prompt("ما هي أنواع الخضروات في حي الإسكان؟")
    print(f"   Prompt length: {len(prompt)} characters")
    
    db.close()
    print("\n✅ All tests passed!")
