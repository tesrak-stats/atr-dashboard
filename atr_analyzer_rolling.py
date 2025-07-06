import streamlit as st
import subprocess
import os
import pandas as pd
import tempfile
import time
from datetime import datetime
import sys

st.set_page_config(page_title="ATR Analyzer - Web & Local", page_icon="🚀", layout="wide")

st.title('🚀 ATR Analyzer - Memory Efficient Edition')
st.write('**Choose your analysis mode based on dataset size and requirements**')

# Check if local analyzer exists
local_analyzer_path = "local_atr_analyzer.py"
if not os.path.exists(local_analyzer_path):
    st.error(f"❌ Local analyzer not found: {local_analyzer_path}")
    st.info("Please save the local_atr_analyzer.py file in the same directory as this Streamlit app.")
    st.stop()

st.success(f"✅ Local analyzer found: {local_analyzer_path}")

# Analysis Mode Selection
st.header("🎯 Analysis Mode Selection")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🚀 Light Mode")
    st.write("**Trigger/Goal Analysis**")
    st.write("• Session analysis")
    st.write("• Rolling analysis")
    st.write("• Memory efficient")
    st.write("• Fast processing")
    st.info("💡 Perfect for web runs and quick analysis")

with col2:
    st.subheader("🔍 Heavy Mode") 
    st.write("**Zone Analysis**")
    st.write("• ZoneBaseline mapping")
    st.write("• StateCheck behavior")
    st.write("• Memory intensive")
    st.write("• Detailed analysis")
    st.warning("⚠️ Better for local runs on large datasets")

with col3:
    st.subheader("💪 Full Mode")
    st.write("**Complete Analysis**")
    st.write("• All analysis types")
    st.write("• Maximum insight")
    st.write("• Highest memory usage")
    st.write("• Longest processing")
    st.error("🚨 Local execution recommended")

# Mode selection
analysis_mode = st.selectbox(
    "Select Analysis Mode",
    ["Light Mode (Trigger/Goal)", "Heavy Mode (Zone Analysis)", "Full Mode (All Analysis)"],
    help="Choose based on your dataset size and available resources"
)

# File upload
st.header("📁 Data Upload")

uploaded_method = st.radio(
    "Choose file input method:",
    ["📁 Browse Local Files", "⬆️ Upload File"],
    help="Browse files directly from your system or upload via web interface"
)

data_file_path = None

if uploaded_method == "📁 Browse Local Files":
    data_file_path = st.text_input(
        "Enter file path:",
        placeholder="/path/to/your/data.csv",
        help="Enter the full path to your pre-formatted CSV file"
    )
    
    if data_file_path and os.path.exists(data_file_path):
        st.success(f"✅ File found: {data_file_path}")
        try:
            file_size = os.path.getsize(data_file_path)
            file_size_mb = file_size / (1024 * 1024)
            st.info(f"📊 File size: {file_size_mb:.1f} MB")
            
            # Size-based recommendations
            if file_size_mb > 100:
                st.warning("⚠️ Large file detected. Consider Light Mode or local execution.")
            elif file_size_mb > 50:
                st.info("💡 Medium file size. Light Mode recommended for web execution.")
        except:
            pass
    elif data_file_path:
        st.error(f"❌ File not found: {data_file_path}")
        
else:
    data_file = st.file_uploader(
        "Upload Pre-formatted Data File",
        type=['csv'],
        help="CSV file processed by the CSV Data Handler"
    )
    
    if data_file:
        st.success(f"✅ Data file uploaded: {data_file.name}")
        
        # Check file size
        file_size = len(data_file.getvalue())
        file_size_mb = file_size / (1024 * 1024)
        st.info(f"📊 File size: {file_size_mb:.1f} MB")
        
        # Size-based recommendations
        if file_size_mb > 100:
            st.warning("⚠️ Large file detected. Consider Light Mode or local execution.")
        elif file_size_mb > 50:
            st.info("💡 Medium file size. Light Mode recommended for web execution.")
        
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as tmp_file:
            tmp_file.write(data_file.getvalue())
            data_file_path = tmp_file.name
        
        st.info(f"📁 Temporary file created: {data_file_path}")

# Configuration section
if data_file_path:
    st.header("⚙️ Analysis Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏷️ Basic Settings")
        
        ticker = st.text_input("Ticker Symbol", value="ES", help="Ticker for file naming")
        asset_type = st.selectbox("Asset Class", ['STOCKS', 'CRYPTO', 'FOREX', 'FUTURES'])
        extended_hours = False
        if asset_type == 'STOCKS':
            extended_hours = st.checkbox("Extended Hours", help="Include pre/after market")
        
        output_dir = st.text_input("Output Directory", value="output", help="Where to save results")
    
    with col2:
        st.subheader("🔧 Processing Options")
        
        # Adjust default batch size based on mode
        if "Light Mode" in analysis_mode:
            default_batch = 50
            batch_help = "Light mode: larger batches OK"
        elif "Heavy Mode" in analysis_mode:
            default_batch = 25
            batch_help = "Heavy mode: smaller batches recommended"
        else:
            default_batch = 15
            batch_help = "Full mode: small batches to prevent memory issues"
        
        progress_interval = st.number_input(
            "Batch Size", 
            min_value=5, 
            max_value=100, 
            value=default_batch,
            help=batch_help
        )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            start_period = st.number_input("Start Period", min_value=0, value=0, help="For resuming analysis")
            end_period = st.number_input("End Period", min_value=0, value=0, help="0 = process all")
            resume_file = st.text_input("Resume File", help="Path to existing results file to resume from")
    
    # Build command based on analysis mode
    cmd_args = [
        data_file_path,
        "--ticker", ticker,
        "--asset_type", asset_type,
        "--output_dir", output_dir,
        "--progress_interval", str(progress_interval)
    ]
    
    # Add analysis mode parameter
    if "Light Mode" in analysis_mode:
        cmd_args.extend(["--analysis_mode", "light"])
    elif "Heavy Mode" in analysis_mode:
        cmd_args.extend(["--analysis_mode", "heavy"])
    else:
        cmd_args.extend(["--analysis_mode", "full"])
    
    if extended_hours:
        cmd_args.append("--extended_hours")
    
    if start_period > 0:
        cmd_args.extend(["--start_period", str(start_period)])
    
    if end_period > 0:
        cmd_args.extend(["--end_period", str(end_period)])
    
    if resume_file:
        cmd_args.extend(["--resume_file", resume_file])
    
    # Show command
    st.subheader("🖥️ Command Preview")
    cmd_display = f"python {local_analyzer_path} " + " ".join(cmd_args)
    st.code(cmd_display, language="bash")
    
    # Analysis mode explanation
    st.subheader("📋 Selected Analysis Details")
    if "Light Mode" in analysis_mode:
        st.info("""
        **Light Mode - Trigger/Goal Analysis**
        - ✅ Session analysis (trigger/goal detection within trading sessions)
        - ✅ Rolling analysis (trigger/goal detection within rolling windows)
        - ✅ Day offset and timing calculations
        - ✅ Memory efficient processing
        - ❌ ZoneBaseline mapping (run separately if needed)
        - ❌ StateCheck behavior analysis (run separately if needed)
        """)
    elif "Heavy Mode" in analysis_mode:
        st.info("""
        **Heavy Mode - Zone Analysis**
        - ❌ Session analysis (run separately if needed)
        - ❌ Rolling analysis (run separately if needed)
        - ✅ ZoneBaseline mapping (price zone crossing analysis)
        - ✅ StateCheck behavior analysis (zone frequency patterns)
        - ⚠️ Memory intensive - monitor system resources
        """)
    else:
        st.info("""
        **Full Mode - Complete Analysis**
        - ✅ Session analysis
        - ✅ Rolling analysis  
        - ✅ ZoneBaseline mapping
        - ✅ StateCheck behavior analysis
        - 🚨 Maximum memory usage - local execution recommended
        """)
    
    # Launch section
    st.header("🚀 Launch Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**🎯 Mode Benefits:**")
        if "Light Mode" in analysis_mode:
            st.write("✅ Web-friendly processing")
            st.write("✅ Fast execution")
            st.write("✅ Low memory usage")
            st.write("✅ Core trigger analysis")
        elif "Heavy Mode" in analysis_mode:
            st.write("✅ Detailed zone mapping")
            st.write("✅ Behavioral analysis")
            st.write("⚠️ Higher memory usage")
            st.write("⚠️ Longer processing time")
        else:
            st.write("✅ Complete analysis")
            st.write("✅ Maximum insight")
            st.write("🚨 Highest resource usage")
            st.write("🚨 Local execution recommended")
    
    with col2:
        st.write("**📊 Expected Output:**")
        if "Light Mode" in analysis_mode:
            st.write("• Session trigger/goal results")
            st.write("• Rolling trigger/goal results")
            st.write("• Timing and offset data")
            st.write("• Reduced file size")
        elif "Heavy Mode" in analysis_mode:
            st.write("• Zone crossing mappings")
            st.write("• State behavior patterns")
            st.write("• Zone frequency analysis")
            st.write("• Large result files")
        else:
            st.write("• All analysis types")
            st.write("• Complete market behavior")
            st.write("• Maximum data coverage")
            st.write("• Largest result files")
    
    # Launch button
    if st.button("🚀 Launch Analysis", type="primary", use_container_width=True):
        
        # Create output directory
        try:
            os.makedirs(output_dir, exist_ok=True)
            st.success(f"📁 Output directory ready: {output_dir}")
        except Exception as e:
            st.error(f"Failed to create output directory: {e}")
            st.stop()
        
        st.info(f"🚀 Launching {analysis_mode.split(' ')[0]} analysis...")
        st.write("**Monitor progress in your terminal/console and in the output directory.**")
        
        # Create placeholders for updates
        status_placeholder = st.empty()
        output_placeholder = st.empty()
        files_placeholder = st.empty()
        
        try:
            # Build full command
            full_cmd = [sys.executable, local_analyzer_path] + cmd_args
            
            # Start the process
            status_placeholder.info("🔄 Starting analysis process...")
            
            process = subprocess.Popen(
                full_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=dict(os.environ, PYTHONUNBUFFERED='1')
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
                    status_placeholder.info(f"🔄 Running... {output.strip()}")
                    
                    # Show recent output (last 20 lines)
                    recent_output = output_lines[-20:] if len(output_lines) > 20 else output_lines
                    output_text = "\n".join(recent_output)
                    output_placeholder.text_area("📋 Live Output", output_text, height=400)
                    
                    # Check for batch files
                    if os.path.exists(output_dir):
                        batch_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
                        if batch_files:
                            files_placeholder.write(f"📁 Files created: {len(batch_files)} batch files")
                    
                    # Small delay to prevent overwhelming
                    time.sleep(0.1)
            
            # Get final return code
            return_code = process.poll()
            
            if return_code == 0:
                status_placeholder.success("✅ Analysis completed successfully!")
                
                # Show output files
                if os.path.exists(output_dir):
                    output_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
                    
                    if output_files:
                        st.header("📁 Results")
                        
                        # Sort files (final results first)
                        final_files = [f for f in output_files if 'FINAL' in f]
                        batch_files = [f for f in output_files if 'BATCH' in f]
                        
                        # Show final results first
                        if final_files:
                            st.subheader("🎯 Final Results")
                            for file in final_files:
                                file_path = os.path.join(output_dir, file)
                                file_size = os.path.getsize(file_path)
                                
                                col1, col2, col3 = st.columns([3, 1, 1])
                                
                                with col1:
                                    st.write(f"📄 **{file}**")
                                
                                with col2:
                                    st.write(f"{file_size:,} bytes")
                                
                                with col3:
                                    try:
                                        with open(file_path, 'rb') as f:
                                            st.download_button(
                                                "⬇️ Download",
                                                data=f.read(),
                                                file_name=file,
                                                mime='text/csv',
                                                key=f"download_final_{file}"
                                            )
                                    except Exception as e:
                                        st.error(f"Error: {e}")
                        
                        # Show batch files if any remain
                        if batch_files:
                            with st.expander(f"📊 Batch Files ({len(batch_files)} files)"):
                                st.info("Note: Batch files are normally cleaned up after combination. These may be leftover from an interrupted run.")
                                for file in sorted(batch_files):
                                    file_path = os.path.join(output_dir, file)
                                    file_size = os.path.getsize(file_path)
                                    
                                    col1, col2, col3 = st.columns([3, 1, 1])
                                    
                                    with col1:
                                        st.write(f"📄 {file}")
                                    
                                    with col2:
                                        st.write(f"{file_size:,} bytes")
                                    
                                    with col3:
                                        try:
                                            with open(file_path, 'rb') as f:
                                                st.download_button(
                                                    "⬇️",
                                                    data=f.read(),
                                                    file_name=file,
                                                    mime='text/csv',
                                                    key=f"download_batch_{file}"
                                                )
                                        except:
                                            st.write("—")
                        
                        # Analyze final results
                        if final_files:
                            largest_file = max(final_files, key=lambda f: os.path.getsize(os.path.join(output_dir, f)))
                            largest_path = os.path.join(output_dir, largest_file)
                            
                            try:
                                # Sample the file for analysis
                                df_sample = pd.read_csv(largest_path, nrows=1000)
                                
                                # Count total rows more efficiently
                                with open(largest_path, 'r') as f:
                                    total_rows = sum(1 for line in f) - 1  # Subtract header
                                
                                st.subheader(f"📊 Analysis Summary: {largest_file}")
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Total Records", f"{total_rows:,}")
                                with col2:
                                    if 'Date' in df_sample.columns:
                                        st.metric("Date Range", f"{df_sample['Date'].nunique()} days")
                                with col3:
                                    if 'AnalysisType' in df_sample.columns:
                                        analysis_types = df_sample['AnalysisType'].unique()
                                        st.metric("Analysis Types", len(analysis_types))
                                with col4:
                                    if 'GoalHit' in df_sample.columns:
                                        hit_rate = (df_sample['GoalHit'] == 'Yes').sum() / len(df_sample) * 100
                                        st.metric("Sample Hit Rate", f"{hit_rate:.1f}%")
                                
                                # Analysis type breakdown
                                if 'AnalysisType' in df_sample.columns:
                                    st.subheader("📈 Analysis Breakdown (Sample)")
                                    analysis_counts = df_sample['AnalysisType'].value_counts()
                                    st.bar_chart(analysis_counts)
                                
                                # Show preview
                                st.subheader("👀 Data Preview")
                                st.dataframe(df_sample.head(10), use_container_width=True)
                                
                            except Exception as e:
                                st.warning(f"Could not analyze results file: {e}")
                    
                    else:
                        st.warning("No CSV files found in output directory")
                else:
                    st.warning(f"Output directory not found: {output_dir}")
            
            else:
                status_placeholder.error(f"❌ Analysis failed with return code: {return_code}")
                st.write("**Check the output above for error details.**")
        
        except Exception as e:
            st.error(f"❌ Error launching analysis: {e}")
        
        finally:
            # Clean up temp file if it was uploaded
            if uploaded_method == "⬆️ Upload File" and data_file_path and data_file_path.startswith("/tmp"):
                try:
                    os.unlink(data_file_path)
                except:
                    pass

else:
    st.info("👆 **Please specify a data file to get started**")
    
    with st.expander("📋 Analysis Mode Guide", expanded=True):
        st.markdown("""
        **🎯 Choose Your Analysis Mode:**
        
        **🚀 Light Mode (Trigger/Goal Analysis)**
        - Perfect for web execution and quick analysis
        - Processes Session and Rolling analysis together
        - Memory efficient with faster processing
        - Great for initial market behavior insights
        - Recommended for files under 100MB
        
        **🔍 Heavy Mode (Zone Analysis)**  
        - Detailed zone mapping and behavioral analysis
        - Processes ZoneBaseline and StateCheck together
        - Memory intensive but provides deep insights
        - Better suited for local execution
        - Recommended for comprehensive zone studies
        
        **💪 Full Mode (Complete Analysis)**
        - All analysis types in one run
        - Maximum insight but highest resource usage
        - Strongly recommended for local execution only
        - Perfect for final comprehensive analysis
        
        **💡 Pro Tip:** Run Light Mode first to get core insights quickly, then run Heavy Mode separately if you need detailed zone analysis. The summarizer can combine both results!
        """)
    
    with st.expander("🔧 Memory Management", expanded=False):
        st.markdown("""
        **📊 Memory Efficiency Improvements:**
        
        ✅ **Batch Processing** - Data processed in small chunks
        ✅ **Progressive File Writing** - Results saved as batches
        ✅ **Automatic Cleanup** - Temporary files removed after combination
        ✅ **Memory Clearing** - Each batch clears previous data
        ✅ **Mode-Based Optimization** - Different strategies per analysis type
        
        **🎯 Batch Size Recommendations:**
        - Light Mode: 50+ periods (efficient trigger processing)
        - Heavy Mode: 25 periods (manages zone analysis memory)
        - Full Mode: 15 periods (prevents memory overload)
        
        **🔄 Recovery Features:**
        - Resume from partial runs
        - Automatic batch file combination
        - Error recovery with saved progress
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🚀 <strong>ATR Analyzer - Memory Efficient Edition</strong></p>
    <p>💡 Smart analysis mode selection for optimal performance</p>
    <p>🔧 Choose Light Mode for web runs, Heavy Mode for detailed analysis</p>
</div>
""", unsafe_allow_html=True)
