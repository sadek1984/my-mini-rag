"""
🔄 LARS Database Migration Script
==================================
Converts Excel files to DuckDB database (one-time migration)

Usage:
    python migrate_db.py

This script:
1. Reads the existing Excel file
2. Creates a DuckDB database file (lars_data.duckdb)
3. Creates a 'samples' table with all the data
4. Verifies the migration was successful

Author: LARS Development Team
Date: 2025-12-12
"""

import duckdb
import pandas as pd
from pathlib import Path
import sys

def migrate_excel_to_duckdb():
    """Convert Excel file to DuckDB database"""
    
    # Configuration (files are in ../data/ folder)
    excel_file = Path(__file__).parent.parent / 'data' / 'processed_data_with_neighborhood.xlsx'
    db_file = Path(__file__).parent.parent / 'data' / 'lars_data.duckdb'
    
    print("=" * 60)
    print("🔄 LARS Database Migration: Excel → DuckDB")
    print("=" * 60)
    
    # Check if Excel file exists
    if not excel_file.exists():
        print(f"❌ Error: Excel file not found at {excel_file}")
        sys.exit(1)
    
    print(f"\n📂 Source: {excel_file}")
    print(f"📦 Target: {db_file}")
    
    # Check if database already exists
    if db_file.exists():
        response = input("\n⚠️ Database already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("❌ Migration cancelled.")
            sys.exit(0)
        db_file.unlink()  # Delete existing file
    
    try:
        # Step 1: Read Excel file
        print("\n📖 Step 1: Reading Excel file...")
        df = pd.read_excel(excel_file)
        print(f"   ✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
        
        # Step 2: Create DuckDB connection (creates file automatically)
        print("\n🗄️ Step 2: Creating DuckDB database...")
        con = duckdb.connect(str(db_file))
        
        # Step 3: Create table and insert data
        print("\n📥 Step 3: Importing data into 'samples' table...")
        con.execute("CREATE TABLE samples AS SELECT * FROM df")
        
        # Step 4: Create indexes for faster queries
        print("\n⚡ Step 4: Creating indexes for better performance...")
        
        # Index on commonly queried columns
        index_columns = [
            'year',
            'vegetable_english',
            'pesticide_standardized',
            'neighborhood_arabic',
            'is_compliant'
        ]
        
        for col in index_columns:
            if col in df.columns:
                try:
                    con.execute(f"CREATE INDEX idx_{col} ON samples({col})")
                    print(f"   ✅ Index created: idx_{col}")
                except Exception as e:
                    print(f"   ⚠️ Could not create index on {col}: {e}")
        
        # Step 5: Verify migration
        print("\n✅ Step 5: Verifying migration...")
        result = con.execute("SELECT COUNT(*) as count FROM samples").fetchone()
        db_count = result[0]
        
        if db_count == len(df):
            print(f"   ✅ Verification passed: {db_count:,} rows in database")
        else:
            print(f"   ⚠️ Warning: Row count mismatch (Excel: {len(df)}, DB: {db_count})")
        
        # Show table schema
        print("\n📋 Table Schema:")
        schema = con.execute("DESCRIBE samples").fetchall()
        for col_name, col_type, *_ in schema:
            print(f"   - {col_name}: {col_type}")
        
        # Close connection
        con.close()
        
        print("\n" + "=" * 60)
        print("🎉 Migration completed successfully!")
        print("=" * 60)
        print(f"\n📦 Database file: {db_file}")
        print(f"📊 Total records: {db_count:,}")
        print(f"💾 File size: {db_file.stat().st_size / 1024:.2f} KB")
        print("\n💡 Next steps:")
        print("   1. Update app.py to use DuckDB")
        print("   2. Test queries with: SELECT * FROM samples LIMIT 10")
        print("   3. Start your Streamlit app")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def migrate_chemistry_sheet():
    """
    Migrate the Chemistry sheet (الكيمياء) from bur_dataset.xlsx to DuckDB
    This adds a 'chemistry' table to the existing database
    """
    
    # Configuration (files are in ../data/ folder)
    excel_file = Path(__file__).parent.parent / 'data' / 'bur_dataset.xlsx'
    db_file = Path(__file__).parent.parent / 'data' / 'lars_data.duckdb'
    sheet_name = "الكيمياء"  # Chemistry sheet in Arabic
    
    print("=" * 60)
    print("🧪 LARS Chemistry Sheet Migration")
    print("=" * 60)
    
    # Check if Excel file exists
    if not excel_file.exists():
        print(f"❌ Error: Excel file not found at {excel_file}")
        sys.exit(1)
    
    print(f"\n📂 Source: {excel_file}")
    print(f"📄 Sheet: {sheet_name}")
    print(f"📦 Target: {db_file}")
    
    try:
        # Step 1: Read Chemistry sheet from Excel
        print(f"\n📖 Step 1: Reading '{sheet_name}' sheet...")
        df_chemistry = pd.read_excel(excel_file, sheet_name=sheet_name)
        print(f"   ✅ Loaded {len(df_chemistry):,} rows, {len(df_chemistry.columns)} columns")
        
        # Show columns
        print(f"\n📋 Columns found:")
        for col in df_chemistry.columns:
            print(f"   - {col}")
        
        # Step 2: Connect to existing DuckDB (or create new)
        print("\n🗄️ Step 2: Connecting to DuckDB database...")
        con = duckdb.connect(str(db_file))
        
        # Drop existing chemistry table if exists
        con.execute("DROP TABLE IF EXISTS chemistry")
        
        # Step 3: Create table and insert data
        print("\n📥 Step 3: Creating 'chemistry' table...")
        con.execute("CREATE TABLE chemistry AS SELECT * FROM df_chemistry")
        
        # Step 4: Verify migration
        print("\n✅ Step 4: Verifying migration...")
        result = con.execute("SELECT COUNT(*) as count FROM chemistry").fetchone()
        db_count = result[0]
        
        if db_count == len(df_chemistry):
            print(f"   ✅ Verification passed: {db_count:,} rows in 'chemistry' table")
        else:
            print(f"   ⚠️ Warning: Row count mismatch (Excel: {len(df_chemistry)}, DB: {db_count})")
        
        # Show table schema
        print("\n📋 Chemistry Table Schema:")
        schema = con.execute("DESCRIBE chemistry").fetchall()
        for col_name, col_type, *_ in schema:
            print(f"   - {col_name}: {col_type}")
        
        # Close connection
        con.close()
        
        print("\n" + "=" * 60)
        print("🎉 Chemistry sheet migration completed!")
        print("=" * 60)
        print(f"\n📊 Total records: {db_count:,}")
        print("💡 Query with: SELECT * FROM chemistry LIMIT 10")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_database():
    """Test the database with sample queries"""
    
    db_file = Path(__file__).parent.parent / 'data' / 'lars_data.duckdb'
    
    if not db_file.exists():
        print("❌ Database not found. Run migration first.")
        return
    
    print("\n🧪 Testing Database Queries...")
    print("-" * 40)
    
    con = duckdb.connect(str(db_file), read_only=True)
    
    # Test queries for samples table
    tests = [
        ("Total samples records", "SELECT COUNT(*) FROM samples"),
        ("Unique vegetables", "SELECT COUNT(DISTINCT vegetable_english) FROM samples"),
        ("Unique pesticides", "SELECT COUNT(DISTINCT pesticide_standardized) FROM samples"),
        ("Compliance rate", "SELECT ROUND(AVG(is_compliant) * 100, 2) as rate FROM samples"),
        ("Records by year", "SELECT year, COUNT(*) as count FROM samples GROUP BY year ORDER BY year"),
    ]
    
    for name, query in tests:
        try:
            result = con.execute(query).fetchall()
            print(f"✅ {name}: {result}")
        except Exception as e:
            print(f"❌ {name}: {e}")
    
    # Test chemistry table if exists
    print("\n🧪 Testing Chemistry Table...")
    print("-" * 40)
    try:
        result = con.execute("SELECT COUNT(*) FROM chemistry").fetchone()
        print(f"✅ Total chemistry records: {result[0]}")
        
        # Show sample result columns
        result = con.execute("SELECT \"نوع العينة\", \"نوع الاختبار\", COUNT(*) as count FROM chemistry GROUP BY \"نوع العينة\", \"نوع الاختبار\" LIMIT 5").fetchall()
        print(f"✅ Sample types and tests: {result}")
    except Exception as e:
        print(f"⚠️ Chemistry table not found or error: {e}")
    
    con.close()
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LARS Database Migration Tool")
    parser.add_argument('--test', action='store_true', help="Test existing database")
    parser.add_argument('--chemistry', action='store_true', help="Migrate Chemistry sheet (الكيمياء) from bur_dataset.xlsx")
    parser.add_argument('--all', action='store_true', help="Migrate all data sources (samples + chemistry)")
    args = parser.parse_args()
    
    if args.test:
        test_database()
    elif args.chemistry:
        migrate_chemistry_sheet()
    elif args.all:
        migrate_excel_to_duckdb()
        migrate_chemistry_sheet()
    else:
        migrate_excel_to_duckdb()

