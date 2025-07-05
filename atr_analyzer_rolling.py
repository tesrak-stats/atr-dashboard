import streamlit as st
import subprocess
import os
import pandas as pd
import tempfile
import time
from datetime import datetime
import sys

st.set_page_config(page_title="ATR Analyzer Launcher", page_icon="🚀", layout="wide")

st.title('🚀 Local ATR Analyzer Launcher')
st.write('**Launch comprehensive ATR analysis locally (no memory limits!)**')

# Check if local analyzer exists
local_analyzer_path = "local_atr_analyzer.py"
if not os.path.exists(local_analyzer_path):
    st.error(f"❌ Local analyzer not found: {local_analyzer_path}")
    st.info("Please save the local_atr_analyzer.py file in the same directory as this Streamlit app.")
    st.stop()

st.success(f"✅ Local analyzer found: {local_analyzer_path}")

# File upload
st.header("📁 Data Upload")
data_file = st.file_uploader(
    "Upload Pre-formatted Data File",
    type=['csv'],
    help="CSV file processed by the CSV Data Handler"
)

if data_file:
    st.success(f"✅ Data file uploaded: {data_file.name}")
    
    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as tmp_file:
        tmp_file.write(data_file.getvalue())
        temp_file_path = tmp_file.name
    
    # Configuration
    st.header("⚙️ Analysis Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        ticker = st.text_input("Ticker Symbol", value="ES", help="Ticker for file naming")
        asset_type = st.selectbox("Asset Class", ['STOCKS', 'CRYPTO', 'FOREX', 'FUTURES'])
        extended_hours = False
        if asset_type == 'STOCKS':
            extended_hours = st.checkbox("Extended Hours", help="Include pre/after market")
    
    with col2:
        output_dir = st.text_input("Output Directory", value="output", help="Where to save results")
        progress_interval = st.number_input("Progress Save Interval", min_value=10, max_value=200, value=50, 
                                          help="Save progress every N periods")
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            start_period = st.number_input("Start Period", min_value=0, value=0, help="For resuming analysis")
            end_period = st.number_input("End Period", min_value=0, value=0, help="0 = process all")
            resume_file = st.text_input("Resume File", help="Path to existing results file to resume from")
    
    # Build command
    cmd = [
        sys.executable, local_analyzer_path,
        temp_file_path,
        "--ticker", ticker,
        "--asset_type", asset_type,
        "--output_dir", output_dir,
        "--progress_interval", str(progress_interval)
    ]
    
    if extended_hours:
        cmd.append("--extended_hours")
    
    if start_period > 0:
        cmd.extend(["--start_period", str(start_period)])
    
    if end_period > 0:
        cmd.extend(["--end_period", str(end_period)])
    
    if resume_file:
        cmd.extend(["--resume_file", resume_file])
    
    # Show command
    st.subheader("🖥️ Command to Execute")
    st.code(" ".join(cmd))
    
    # Launch button
    if st.button("🚀 Launch Local Analysis", type="primary", use_container_width=True):
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        st.info("🚀 Launching local analysis...")
        st.write("**This will run in the background. Check the terminal/console for progress updates.**")
        
        # Create a placeholder for status updates
        status_placeholder = st.empty()
        output_placeholder = st.empty()
        
        try:
            # Start the process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Monitor the process
            output_lines = []
            
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    output_lines.append(output.strip())
                    
                    # Update status
                    status_placeholder.info(f"🔄 Running... Last output: {output.strip()}")
                    
                    # Show recent output (last 20 lines)
                    if len(output_lines) > 20:
                        recent_output = output_lines[-20:]
                    else:
                        recent_output = output_lines
                    
                    output_text = "\n".join(recent_output)
                    output_placeholder.text_area("📋 Live Output", output_text, height=400)
                    
                    # Force update
                    time.sleep(0.1)
            
            # Get final return code
            return_code = process.poll()
            
            if return_code == 0:
                st.success("✅ Analysis completed successfully!")
                
                # Look for output files
                if os.path.exists(output_dir):
                    output_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
                    
                    if output_files:
                        st.subheader("📁 Output Files")
                        
                        for file in sorted(output_files):
                            file_path = os.path.join(output_dir, file)
                            file_size = os.path.getsize(file_path)
                            
                            col1, col2, col3 = st.columns([3, 1, 1])
                            
                            with col1:
                                st.write(f"📄 **{file}**")
                            
                            with col2:
                                st.write(f"{file_size:,} bytes")
                            
                            with col3:
                                # Read and offer download
                                try:
                                    with open(file_path, 'rb') as f:
                                        st.download_button(
                                            "⬇️ Download",
                                            data=f.read(),
                                            file_name=file,
                                            mime='text/csv',
                                            key=f"download_{file}"
                                        )
                                except Exception as e:
                                    st.error(f"Error reading {file}: {e}")
                        
                        # Show summary of largest file (likely the final results)
                        largest_file = max(output_files, key=lambda f: os.path.getsize(os.path.join(output_dir, f)))
                        largest_path = os.path.join(output_dir, largest_file)
                        
                        try:
                            df = pd.read_csv(largest_path)
                            
                            st.subheader(f"📊 Summary: {largest_file}")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Records", f"{len(df):,}")
                            with col2:
                                if 'Date' in df.columns:
                                    st.metric("Unique Dates", df['Date'].nunique())
                            with col3:
                                if 'AnalysisType' in df.columns:
                                    st.metric("Analysis Types", df['AnalysisType'].nunique())
                            with col4:
                                if 'GoalHit' in df.columns:
                                    hit_rate = (df['GoalHit'] == 'Yes').sum() / len(df) * 100
                                    st.metric("Hit Rate", f"{hit_rate:.1f}%")
                            
                            # Analysis type breakdown
                            if 'AnalysisType' in df.columns:
                                st.subheader("📈 Analysis Breakdown")
                                analysis_counts = df['AnalysisType'].value_counts()
                                st.bar_chart(analysis_counts)
                            
                            # Show preview
                            st.subheader("👀 Data Preview")
                            st.dataframe(df.head(10), use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Error analyzing results: {e}")
                    
                    else:
                        st.warning("No CSV files found in output directory")
                else:
                    st.warning(f"Output directory not found: {output_dir}")
            
            else:
                st.error(f"❌ Analysis failed with return code: {return_code}")
                st.write("**Check the output above for error details.**")
        
        except Exception as e:
            st.error(f"❌ Error launching analysis: {e}")
        
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file_path)
            except:
                pass

else:
    st.info("👆 **Please upload a pre-formatted CSV file to get started**")
    
    with st.expander("📋 Local Analysis Benefits", expanded=True):
        st.markdown("""
        **🚀 Why Run Locally?**
        
        ✅ **No Memory Limits** - Process massive datasets without Streamlit constraints
        ✅ **No Timeouts** - Run for hours without interruption  
        ✅ **Better Performance** - Use your full system resources
        ✅ **Progressive Saving** - Results saved as you go (no lost work)
        ✅ **Resume Capability** - Continue from where you left off
        ✅ **File Management** - Direct access to output files
        
        **🎯 Perfect For:**
        - Large datasets (18+ years of data)
        - Memory-intensive analysis (ZoneBaseline + StateCheck)
        - Long-running comprehensive analysis
        - Production data processing
        
        **📁 Output:**
        - Progress files saved every 50 periods
        - Final comprehensive results file
        - All 4 analysis types included
        - Complete metadata for downstream apps
        """)
    
    with st.expander("🔧 Setup Instructions", expanded=False):
        st.markdown("""
        **📋 Setup Steps:**
        
        1. **Save Local Analyzer:**
           - Copy the `local_atr_analyzer.py` code 
           - Save in same directory as this Streamlit app
        
        2. **Install Dependencies:**
           ```bash
           pip install pandas numpy
           ```
        
        3. **Prepare Data:**
           - Use CSV Data Handler to prepare your data file
           - Ensure all required columns are present
        
        4. **Launch Analysis:**
           - Upload your CSV file
           - Configure settings
           - Click "Launch Local Analysis"
        
        5. **Monitor Progress:**
           - Watch live output in the terminal
           - Progress files saved automatically
           - Download results when complete
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🎯 <strong>Local ATR Analyzer Launcher</strong> - Unlimited processing power for comprehensive market analysis</p>
</div>
""", unsafe_allow_html=True)
