import streamlit as st
import pandas as pd
import numpy as np

def calculate_atr(df, period=14):
    """
    Calculate Wilder's ATR (Average True Range)
    """
    df = df.copy()
    
    # Calculate True Range
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    
    # Calculate Wilder's ATR (exponential moving average with alpha = 1/period)
    df['ATR_Calculated'] = df['TR'].ewm(alpha=1/period, adjust=False).mean()
    
    # Clean up temporary columns
    df.drop(['H-L', 'H-PC', 'L-PC', 'TR'], axis=1, inplace=True)
    
    return df

def generate_atr_levels(close_price, atr_value):
    """
    Generate Fibonacci-based ATR levels from the prior day's close
    """
    fib_ratios = [1.000, 0.786, 0.618, 0.500, 0.382, 0.236, 0.000, 
                  -0.236, -0.382, -0.500, -0.618, -0.786, -1.000]
    
    levels = {}
    for ratio in fib_ratios:
        level_price = close_price + (ratio * atr_value)
        levels[f'Level_{ratio}'] = level_price
    
    return levels

def compare_levels(calculated_level, excel_level, tolerance=0.01):
    """
    Compare calculated level with Excel level
    """
    if pd.isna(excel_level) or pd.isna(calculated_level):
        return 'MISSING'
    
    diff = abs(calculated_level - excel_level)
    if diff <= tolerance:
        return 'MATCH'
    else:
        return f'DIFF_{diff:.3f}'

def test_atr_calculation():
    """
    Test ATR calculation and level generation against Excel values
    """
    debug_info = []
    
    try:
        debug_info.append("📊 Loading daily data...")
        daily = pd.read_excel('SPXdailycandles.xlsx', header=4)
        debug_info.append(f"Daily data shape: {daily.shape}")
        debug_info.append(f"Available columns: {list(daily.columns)}")
        
        # Check if we have required OHLC columns
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in daily.columns]
        
        if missing_cols:
            debug_info.append(f"❌ Missing required columns: {missing_cols}")
            return pd.DataFrame(), debug_info
        
        debug_info.append("🧮 Calculating ATR...")
        daily = calculate_atr(daily, period=14)
        
        # Try to find existing ATR column in Excel
        atr_cols = [col for col in daily.columns if 'ATR' in str(col).upper()]
        debug_info.append(f"Found ATR-related columns: {atr_cols}")
        
        # Create comparison DataFrame
        comparison_results = []
        
        # Test on last 20 rows (to have good ATR values)
        test_rows = daily.tail(20).copy()
        
        for idx, row in test_rows.iterrows():
            if idx == 0:  # Skip first row (no previous close)
                continue
                
            date = row['Date']
            current_close = row['Close']
            calculated_atr = row['ATR_Calculated']
            
            # Get previous row for level calculation
            prev_idx = daily.index[daily.index.get_loc(idx) - 1]
            prev_row = daily.loc[prev_idx]
            prev_close = prev_row['Close']
            prev_atr_calc = prev_row['ATR_Calculated']
            
            if pd.isna(prev_atr_calc):
                continue
            
            # Generate calculated levels
            calculated_levels = generate_atr_levels(prev_close, prev_atr_calc)
            
            # Try to find Excel ATR value
            excel_atr = None
            for atr_col in atr_cols:
                if not pd.isna(prev_row[atr_col]):
                    excel_atr = prev_row[atr_col]
                    break
            
            # Create comparison row
            comp_row = {
                'Date': date,
                'PrevClose': prev_close,
                'ATR_Calculated': prev_atr_calc,
                'ATR_Excel': excel_atr,
                'ATR_Match': compare_levels(prev_atr_calc, excel_atr, tolerance=0.01) if excel_atr else 'NO_EXCEL_ATR'
            }
            
            # Add calculated levels
            for level_name, level_value in calculated_levels.items():
                comp_row[f'Calc_{level_name}'] = level_value
            
            # Try to match with Excel levels
            fib_ratios = [1.000, 0.786, 0.618, 0.500, 0.382, 0.236, 0.000, 
                         -0.236, -0.382, -0.500, -0.618, -0.786, -1.000]
            
            for ratio in fib_ratios:
                # Try different possible column names for this ratio
                possible_excel_cols = [
                    str(ratio),
                    str(int(ratio)) if ratio == int(ratio) else str(ratio),
                    f'{ratio:.1f}' if ratio != int(ratio) else str(int(ratio)),
                    f'{ratio:.3f}'.rstrip('0').rstrip('.')
                ]
                
                excel_level = None
                found_col = None
                
                for col_name in possible_excel_cols:
                    if col_name in row.index and not pd.isna(row[col_name]):
                        excel_level = row[col_name]
                        found_col = col_name
                        break
                
                # Add Excel level and comparison
                comp_row[f'Excel_Level_{ratio}'] = excel_level
                comp_row[f'Excel_Col_{ratio}'] = found_col
                
                if excel_level is not None:
                    calc_level = calculated_levels[f'Level_{ratio}']
                    comp_row[f'Match_{ratio}'] = compare_levels(calc_level, excel_level, tolerance=0.05)
                    comp_row[f'Diff_{ratio}'] = abs(calc_level - excel_level) if not pd.isna(excel_level) else None
                else:
                    comp_row[f'Match_{ratio}'] = 'NO_EXCEL_LEVEL'
                    comp_row[f'Diff_{ratio}'] = None
            
            comparison_results.append(comp_row)
        
        comparison_df = pd.DataFrame(comparison_results)
        debug_info.append(f"✅ Generated {len(comparison_df)} comparison rows")
        
        # Summary statistics
        if not comparison_df.empty:
            # Count matches across all levels
            match_cols = [col for col in comparison_df.columns if col.startswith('Match_') and col != 'ATR_Match']
            total_comparisons = 0
            total_matches = 0
            
            for col in match_cols:
                valid_comparisons = comparison_df[col].notna() & (comparison_df[col] != 'NO_EXCEL_LEVEL')
                total_comparisons += valid_comparisons.sum()
                total_matches += (comparison_df[col] == 'MATCH').sum()
            
            if total_comparisons > 0:
                match_rate = (total_matches / total_comparisons) * 100
                debug_info.append(f"📈 Level Match Rate: {match_rate:.1f}% ({total_matches}/{total_comparisons})")
            
            # ATR comparison
            atr_matches = (comparison_df['ATR_Match'] == 'MATCH').sum()
            atr_total = (comparison_df['ATR_Match'] != 'NO_EXCEL_ATR').sum()
            if atr_total > 0:
                atr_match_rate = (atr_matches / atr_total) * 100
                debug_info.append(f"🎯 ATR Match Rate: {atr_match_rate:.1f}% ({atr_matches}/{atr_total})")
        
        return comparison_df, debug_info
        
    except Exception as e:
        debug_info.append(f"❌ Error: {str(e)}")
        import traceback
        debug_info.append(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame(), debug_info

# Streamlit Interface
st.title('🧪 ATR Calculation Validation Test')
st.write('Tests our ATR calculations against your Excel values')

if st.button('🔍 Run ATR Validation Test'):
    with st.spinner('Testing ATR calculations...'):
        try:
            result_df, debug_messages = test_atr_calculation()
            
            # Show debug info
            st.subheader('📋 Debug Information')
            for msg in debug_messages:
                st.write(msg)
            
            if not result_df.empty:
                # Save results
                output_file = 'atr_validation_results.csv'
                result_df.to_csv(output_file, index=False)
                st.success(f'✅ Validation results saved to {output_file}')
                
                # Show summary
                st.subheader('📊 Validation Summary')
                
                # ATR comparison summary
                if 'ATR_Match' in result_df.columns:
                    atr_summary = result_df['ATR_Match'].value_counts()
                    st.write("**ATR Comparison:**")
                    st.write(atr_summary)
                
                # Level comparison summary
                match_cols = [col for col in result_df.columns if col.startswith('Match_') and col != 'ATR_Match']
                if match_cols:
                    st.write("**Level Comparison Summary:**")
                    for col in match_cols[:5]:  # Show first 5 levels
                        level_name = col.replace('Match_', '')
                        summary = result_df[col].value_counts()
                        st.write(f"Level {level_name}: {summary.to_dict()}")
                
                # Preview results
                st.subheader('🔍 Preview of Validation Results')
                # Show key columns first
                key_cols = ['Date', 'PrevClose', 'ATR_Calculated', 'ATR_Excel', 'ATR_Match']
                display_cols = [col for col in key_cols if col in result_df.columns]
                st.dataframe(result_df[display_cols].head(10))
                
                # Full results
                with st.expander('📄 Full Results (All Columns)'):
                    st.dataframe(result_df)
                
                # Download button
                st.download_button(
                    '⬇️ Download Validation Results', 
                    data=result_df.to_csv(index=False), 
                    file_name=output_file, 
                    mime='text/csv'
                )
                
                st.info('💡 **Next Steps:** If match rates are high (>95%), we can confidently use the calculated levels instead of reading from Excel!')
                
            else:
                st.warning('⚠️ No validation results generated - check debug info above')
                
        except Exception as e:
            st.error(f'❌ Error: {e}')

st.markdown("""
---
**What this test does:**
1. 📊 Reads OHLC data from your Excel file
2. 🧮 Calculates ATR using Wilder's method
3. 🎯 Generates Fibonacci ATR levels
4. 🔍 Compares with existing Excel values
5. 📈 Shows match rates and differences
6. 💾 Saves detailed comparison to CSV
""")
