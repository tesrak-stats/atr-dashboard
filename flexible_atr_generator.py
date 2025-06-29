import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import tempfile
import shutil

class DataProcessingCheckpoint:
    """Handle checkpoint/resume functionality for multi-file processing"""
    
    def __init__(self, session_id=None):
        # Use session-specific checkpoint file
        if session_id is None:
            session_id = st.session_state.get('session_id', self._generate_session_id())
            st.session_state['session_id'] = session_id
        
        self.checkpoint_file = f"checkpoint_{session_id}.json"
        self.temp_dir = f"temp_{session_id}"
        self.state = self.load_checkpoint()
    
    def _generate_session_id(self):
        """Generate unique session ID"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def load_checkpoint(self):
        """Load existing checkpoint or create new one"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    return json.load(f)
            except:
                pass  # If checkpoint is corrupted, start fresh
        
        return {
            'processed_files': [],
            'failed_files': [],
            'last_successful_file': None,
            'target_timeframe': None,
            'start_time': None,
            'total_files': 0,
            'temp_data_files': [],
            'session_id': st.session_state.get('session_id'),
            'processing_complete': False
        }
    
    def save_checkpoint(self):
        """Save current state to disk"""
        # Ensure temp directory exists
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(self.state, f, indent=2, default=str)
        except Exception as e:
            st.error(f"Failed to save checkpoint: {e}")
    
    def mark_file_processed(self, filename, temp_data_file=None, row_count=0):
        """Mark a file as successfully processed"""
        self.state['processed_files'].append({
            'filename': filename,
            'temp_file': temp_data_file,
            'row_count': row_count,
            'timestamp': datetime.now().isoformat()
        })
        self.state['last_successful_file'] = filename
        if temp_data_file:
            self.state['temp_data_files'].append(temp_data_file)
        self.save_checkpoint()
    
    def mark_file_failed(self, filename, error_msg):
        """Mark a file as failed with error details"""
        self.state['failed_files'].append({
            'filename': filename,
            'error': str(error_msg),
            'timestamp': datetime.now().isoformat()
        })
        self.save_checkpoint()
    
    def clear_checkpoint(self):
        """Clear checkpoint and cleanup temp files"""
        # Remove temp files
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Remove checkpoint file
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
        
        # Reset state
        self.state = {
            'processed_files': [],
            'failed_files': [],
            'last_successful_file': None,
            'target_timeframe': None,
            'start_time': None,
            'total_files': 0,
            'temp_data_files': [],
            'session_id': st.session_state.get('session_id'),
            'processing_complete': False
        }
    
    def get_processed_filenames(self):
        """Get list of already processed filenames"""
        return [f['filename'] for f in self.state['processed_files']]
    
    def get_failed_filenames(self):
        """Get list of failed filenames"""
        return [f['filename'] for f in self.state['failed_files']]

class EnhancedMultiFileLoader:
    """Enhanced multi-file loader with resampling and checkpoint support"""
    
    def __init__(self):
        self.checkpoint = DataProcessingCheckpoint()
    
    def standardize_columns(self, df):
        """Standardize column names across different data sources"""
        # Clean column names first
        clean_columns = []
        for col in df.columns:
            if isinstance(col, str):
                clean_columns.append(col.strip())
            else:
                clean_columns.append(str(col).strip())
        
        df.columns = clean_columns
        
        # Common column mappings
        column_mappings = {
            'date': 'Date', 'timestamp': 'Date', 'datetime': 'Datetime',
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
            'last': 'Close', 'adj close': 'Close', 'adjusted_close': 'Close',
            'settle': 'Close', 'volume': 'Volume', 'vol': 'Volume',
            'session': 'Session'
        }
        
        # Apply mappings (case insensitive)
        for old_name, new_name in column_mappings.items():
            for col in df.columns:
                if isinstance(col, str) and col.lower() == old_name:
                    df.rename(columns={col: new_name}, inplace=True)
                    break
        
        return df
    
    def resample_ohlc_data(self, df, timeframe='10T'):
        """Resample OHLC data to specified timeframe"""
        df = df.copy()
        
        # Ensure we have datetime column
        if 'Datetime' not in df.columns:
            if 'Date' in df.columns and 'Time' in df.columns:
                df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
            elif 'Date' in df.columns:
                df['Datetime'] = pd.to_datetime(df['Date'])
            else:
                raise ValueError("Could not find datetime information")
        else:
            df['Datetime'] = pd.to_datetime(df['Datetime'])
        
        # Set datetime as index for resampling
        df.set_index('Datetime', inplace=True)
        
        # Define aggregation rules
        agg_rules = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min', 
            'Close': 'last'
        }
        
        # Add volume if present
        if 'Volume' in df.columns:
            agg_rules['Volume'] = 'sum'
        
        # Add session if present
        if 'Session' in df.columns:
            agg_rules['Session'] = 'first'
        
        # Resample
        resampled = df.resample(timeframe, closed='left', label='left').agg(agg_rules)
        
        # Remove rows with no data (NaN in OHLC)
        resampled = resampled.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        # Reset index to get Datetime back as column
        resampled = resampled.reset_index()
        
        # Create Date column for compatibility
        resampled['Date'] = resampled['Datetime'].dt.date
        
        return resampled
    
    def load_single_file_with_resampling(self, uploaded_file, target_timeframe='10T'):
        """Load and resample a single file"""
        try:
            # Determine file type and load
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file, header=0)
            else:
                raise ValueError(f"Unsupported file format: {uploaded_file.name}")
            
            # Standardize columns
            df = self.standardize_columns(df)
            
            # Validate required columns
            required_cols = ['Open', 'High', 'Low', 'Close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Resample if needed
            if target_timeframe != '1T':  # Don't resample if target is 1 minute
                df_resampled = self.resample_ohlc_data(df, target_timeframe)
            else:
                # Still need to ensure proper datetime handling
                if 'Datetime' not in df.columns:
                    if 'Date' in df.columns and 'Time' in df.columns:
                        df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
                    elif 'Date' in df.columns:
                        df['Datetime'] = pd.to_datetime(df['Date'])
                    else:
                        raise ValueError("Could not find datetime information")
                
                df['Date'] = pd.to_datetime(df['Datetime']).dt.date
                df_resampled = df
            
            return df_resampled
            
        except Exception as e:
            raise Exception(f"Error processing {uploaded_file.name}: {str(e)}")
    
    def load_multiple_files_with_checkpoints(self, uploaded_files, target_timeframe='10T'):
        """Load multiple files with checkpoint support"""
        
        # Check for existing checkpoint
        if self.checkpoint.state['processed_files'] or self.checkpoint.state['failed_files']:
            st.warning("🔄 **Previous Processing Session Found**")
            
            col1, col2 = st.columns(2)
            with col1:
                if self.checkpoint.state['processed_files']:
                    st.success(f"✅ Previously processed: {len(self.checkpoint.state['processed_files'])} files")
                    total_rows = sum(f.get('row_count', 0) for f in self.checkpoint.state['processed_files'])
                    st.info(f"📊 Total processed rows: {total_rows:,}")
            
            with col2:
                if self.checkpoint.state['failed_files']:
                    st.error(f"❌ Previous failures: {len(self.checkpoint.state['failed_files'])} files")
                    with st.expander("Show Failed Files"):
                        for failed in self.checkpoint.state['failed_files']:
                            st.error(f"• **{failed['filename']}**")
                            st.write(f"  Error: {failed['error']}")
                            st.write(f"  Time: {failed['timestamp']}")
            
            # Recovery options
            recovery_choice = st.radio(
                "How would you like to proceed?",
                [
                    "Continue from checkpoint (recommended)",
                    "Start fresh (clear all progress)", 
                    "Skip failed files and continue"
                ],
                key="recovery_choice"
            )
            
            if recovery_choice == "Start fresh (clear all progress)":
                self.checkpoint.clear_checkpoint()
                st.success("✅ Checkpoint cleared. Please re-run to start fresh.")
                return None
            elif recovery_choice == "Skip failed files and continue":
                failed_names = self.checkpoint.get_failed_filenames()
                uploaded_files = [f for f in uploaded_files if f.name not in failed_names]
                st.info(f"📋 Skipping {len(failed_names)} failed files")
        
        # Initialize or update checkpoint
        file_names = [f.name for f in uploaded_files]
        self.checkpoint.state['total_files'] = len(uploaded_files)
        self.checkpoint.state['target_timeframe'] = target_timeframe
        if not self.checkpoint.state['start_time']:
            self.checkpoint.state['start_time'] = datetime.now().isoformat()
        self.checkpoint.save_checkpoint()
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        detail_text = st.empty()
        
        processed_dataframes = []
        processed_filenames = self.checkpoint.get_processed_filenames()
        
        # Process files
        for i, uploaded_file in enumerate(uploaded_files):
            
            # Skip already processed files
            if uploaded_file.name in processed_filenames:
                status_text.text(f"✅ Skipping {uploaded_file.name} (already processed)")
                
                # Load from temp file
                temp_filename = os.path.join(self.checkpoint.temp_dir, f"{uploaded_file.name}_{target_timeframe}.parquet")
                if os.path.exists(temp_filename):
                    try:
                        df_temp = pd.read_parquet(temp_filename)
                        processed_dataframes.append(df_temp)
                        detail_text.text(f"📂 Loaded {len(df_temp):,} rows from cache")
                    except Exception as e:
                        st.warning(f"⚠️ Could not load cached data for {uploaded_file.name}: {e}")
                
                progress_bar.progress((i + 1) / len(uploaded_files))
                continue
            
            try:
                # Update status
                status_text.text(f"🔄 Processing {uploaded_file.name} ({i+1}/{len(uploaded_files)})...")
                detail_text.text(f"📁 Loading file...")
                
                # Process the file
                df_processed = self.load_single_file_with_resampling(uploaded_file, target_timeframe)
                
                detail_text.text(f"📊 Processed {len(df_processed):,} rows")
                
                # Save to temp file
                temp_filename = os.path.join(self.checkpoint.temp_dir, f"{uploaded_file.name}_{target_timeframe}.parquet")
                os.makedirs(self.checkpoint.temp_dir, exist_ok=True)
                df_processed.to_parquet(temp_filename)
                
                # Add to collection
                processed_dataframes.append(df_processed)
                
                # Update checkpoint
                self.checkpoint.mark_file_processed(
                    uploaded_file.name, 
                    temp_filename, 
                    len(df_processed)
                )
                
                # Show success
                st.success(f"✅ {uploaded_file.name}: {len(df_processed):,} bars ({target_timeframe})")
                
            except Exception as e:
                # Handle error
                error_msg = str(e)
                st.error(f"❌ **Error with {uploaded_file.name}**")
                st.error(f"Details: {error_msg}")
                
                self.checkpoint.mark_file_failed(uploaded_file.name, error_msg)
                
                # Give user choice
                error_choice = st.radio(
                    f"How to handle error with {uploaded_file.name}?",
                    ["Skip this file and continue", "Stop processing here"],
                    key=f"error_choice_{i}_{uploaded_file.name}"
                )
                
                if error_choice == "Stop processing here":
                    st.error("🛑 **Processing Stopped**")
                    st.info("💾 Progress has been saved. You can resume by re-running this section.")
                    return None
                else:
                    st.info(f"⏭️ Skipping {uploaded_file.name} and continuing...")
            
            # Update progress
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        # Combine all processed data
        if processed_dataframes:
            status_text.text("🔗 Combining all processed files...")
            detail_text.text("📊 Sorting by datetime...")
            
            # Concatenate and sort
            combined_df = pd.concat(processed_dataframes, ignore_index=True)
            combined_df = combined_df.sort_values('Datetime').reset_index(drop=True)
            
            # Success summary
            total_rows = len(combined_df)
            date_range = f"{combined_df['Date'].min()} to {combined_df['Date'].max()}"
            
            st.success("🎉 **Multi-File Processing Complete!**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📁 Files Processed", len(processed_dataframes))
            with col2:
                st.metric("📊 Total Bars", f"{total_rows:,}")
            with col3:
                st.metric("📅 Date Range", date_range)
            
            # Mark processing as complete
            self.checkpoint.state['processing_complete'] = True
            self.checkpoint.save_checkpoint()
            
            # Cleanup option
            if st.button("🧹 Clean Up Temporary Files"):
                self.checkpoint.clear_checkpoint()
                st.success("✅ Temporary files cleaned up")
            
            return combined_df
        
        else:
            st.warning("⚠️ No files were successfully processed")
            return None

def create_enhanced_multifile_interface():
    """Create the enhanced multi-file upload interface"""
    
    st.title("📁 Enhanced Multi-File ATR Data Processor")
    st.write("Upload multiple data files (e.g., yearly archives) and process them with automatic resampling and checkpoint support")
    
    # File upload section
    st.subheader("📂 File Upload")
    
    uploaded_files = st.file_uploader(
        "Select Multiple Data Files",
        type=['csv', 'xlsx', 'xls'],
        accept_multiple_files=True,
        help="Upload multiple files - they'll be processed sequentially with checkpoint support"
    )
    
    if not uploaded_files:
        st.info("👆 **Please upload your data files to begin processing**")
        st.markdown("""
        ### 💡 **Multi-File Processing Features:**
        - **📁 Batch Processing**: Upload multiple yearly files at once
        - **⏱️ Resampling**: Convert 1-minute data to 5min/10min/15min automatically  
        - **💾 Checkpoint Support**: Resume processing if interrupted
        - **❌ Error Recovery**: Skip problematic files and continue
        - **🔄 Progress Tracking**: See exactly what's been processed
        """)
        return None
    
    # Show uploaded files
    st.success(f"📁 **{len(uploaded_files)} files selected:**")
    
    cols = st.columns(min(len(uploaded_files), 4))
    for i, file in enumerate(uploaded_files):
        with cols[i % 4]:
            file_size_mb = file.size / (1024 * 1024)
            st.write(f"• **{file.name}**")
            st.write(f"  Size: {file_size_mb:.1f} MB")
    
    # Processing configuration
    st.subheader("⚙️ Processing Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        target_timeframe = st.selectbox(
            "Resample to:",
            options=['1T', '3T', '5T', '10T', '15T', '30T', '1H'],
            index=3,  # Default to 10T
            format_func=lambda x: {
                '1T': '1 Minute (No resampling)',
                '3T': '3 Minutes', 
                '5T': '5 Minutes',
                '10T': '10 Minutes',
                '15T': '15 Minutes', 
                '30T': '30 Minutes',
                '1H': '1 Hour'
            }[x],
            help="Target timeframe for resampling. 1-minute data will be aggregated to this timeframe."
        )
    
    with col2:
        st.info("📊 **Estimated Output Size**")
        if target_timeframe == '1T':
            st.write("Same as input")
        else:
            reduction = {'3T': '67%', '5T': '80%', '10T': '90%', '15T': '93%', '30T': '97%', '1H': '98%'}
            st.write(f"~{reduction.get(target_timeframe, 'Unknown')} smaller")
    
    with col3:
        st.info("⚡ **Processing Features**")
        st.write("✅ Checkpoint/Resume")
        st.write("✅ Error Recovery") 
        st.write("✅ Progress Tracking")
    
    # Processing button
    if st.button("🚀 Start Multi-File Processing", type="primary"):
        
        with st.container():
            # Initialize loader
            loader = EnhancedMultiFileLoader()
            
            # Process files
            combined_data = loader.load_multiple_files_with_checkpoints(
                uploaded_files, 
                target_timeframe
            )
            
            if combined_data is not None:
                # Store in session state for further use
                st.session_state['multifile_data'] = combined_data
                st.session_state['multifile_timeframe'] = target_timeframe
                st.session_state['multifile_ready'] = True
                
                # Show preview
                st.subheader("📋 Data Preview")
                
                # Data summary
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 Total Records", f"{len(combined_data):,}")
                with col2:
                    st.metric("📅 Date Range", f"{combined_data['Date'].nunique()} days")
                with col3:
                    unique_hours = combined_data['Datetime'].dt.hour.nunique() if 'Datetime' in combined_data.columns else 'N/A'
                    st.metric("⏰ Hours Covered", unique_hours)
                with col4:
                    file_count = len([f for f in uploaded_files if f.name in loader.checkpoint.get_processed_filenames()] + [f for f in processed_dataframes])
                    st.metric("📁 Files Combined", len(uploaded_files))
                
                # Sample data preview
                st.dataframe(combined_data.head(20), use_container_width=True)
                
                # Data quality check
                with st.expander("📊 Data Quality Summary"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Column Information:**")
                        for col in combined_data.columns:
                            null_count = combined_data[col].isnull().sum()
                            st.write(f"• {col}: {null_count:,} nulls")
                    
                    with col2:
                        st.write("**Data Range:**")
                        if 'Datetime' in combined_data.columns:
                            st.write(f"• First record: {combined_data['Datetime'].min()}")
                            st.write(f"• Last record: {combined_data['Datetime'].max()}")
                            st.write(f"• Time span: {(combined_data['Datetime'].max() - combined_data['Datetime'].min()).days} days")
                        
                        numeric_cols = ['Open', 'High', 'Low', 'Close']
                        for col in numeric_cols:
                            if col in combined_data.columns:
                                st.write(f"• {col} range: {combined_data[col].min():.2f} - {combined_data[col].max():.2f}")
                
                # Export options
                st.subheader("💾 Export Processed Data")
                
                # Generate intelligent filename
                first_date = combined_data['Date'].min().strftime("%Y%m%d") if hasattr(combined_data['Date'].min(), 'strftime') else str(combined_data['Date'].min()).replace('-', '')
                last_date = combined_data['Date'].max().strftime("%Y%m%d") if hasattr(combined_data['Date'].max(), 'strftime') else str(combined_data['Date'].max()).replace('-', '')
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                
                # Default filename suggestion
                default_filename = f"combined_{target_timeframe}_{first_date}_to_{last_date}_{timestamp}.csv"
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    custom_filename = st.text_input(
                        "Output Filename:",
                        value=default_filename,
                        help="Customize the output filename (include .csv extension)"
                    )
                
                with col2:
                    st.write("**File Size Estimate:**")
                    estimated_size_mb = len(combined_data) * len(combined_data.columns) * 10 / (1024 * 1024)  # Rough estimate
                    st.write(f"~{estimated_size_mb:.1f} MB")
                
                # Export options with different formats
                export_col1, export_col2, export_col3 = st.columns(3)
                
                with export_col1:
                    # Standard CSV export
                    csv_data = combined_data.to_csv(index=False)
                    
                    st.download_button(
                        "📥 Download as CSV",
                        data=csv_data,
                        file_name=custom_filename,
                        mime="text/csv",
                        help="Download the complete combined dataset as CSV"
                    )
                
                with export_col2:
                    # Compressed CSV export for large files
                    import gzip
                    csv_compressed = gzip.compress(csv_data.encode('utf-8'))
                    compressed_filename = custom_filename.replace('.csv', '.csv.gz')
                    
                    st.download_button(
                        "📦 Download Compressed",
                        data=csv_compressed,
                        file_name=compressed_filename,
                        mime="application/gzip",
                        help="Download as compressed CSV (smaller file size)"
                    )
                
                with export_col3:
                    # Sample data export (first 1000 rows for testing)
                    sample_data = combined_data.head(1000).to_csv(index=False)
                    sample_filename = custom_filename.replace('.csv', '_sample1000.csv')
                    
                    st.download_button(
                        "📋 Download Sample",
                        data=sample_data,
                        file_name=sample_filename,
                        mime="text/csv",
                        help="Download first 1000 rows for testing"
                    )
                
                # Additional export formats
                with st.expander("🔧 Advanced Export Options"):
                    
                    # Date range filtering for export
                    st.write("**Filter Data for Export:**")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        start_date_filter = st.date_input(
                            "Start Date (optional):",
                            value=None,
                            help="Leave empty to include all data"
                        )
                    
                    with col2:
                        end_date_filter = st.date_input(
                            "End Date (optional):",
                            value=None,
                            help="Leave empty to include all data"
                        )
                    
                    # Column selection for export
                    st.write("**Select Columns to Export:**")
                    all_columns = list(combined_data.columns)
                    
                    # Suggest core columns by default
                    default_columns = []
                    core_cols = ['Datetime', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Session']
                    for col in core_cols:
                        if col in all_columns:
                            default_columns.append(col)
                    
                    selected_columns = st.multiselect(
                        "Columns to include:",
                        options=all_columns,
                        default=default_columns,
                        help="Select which columns to include in the export"
                    )
                    
                    # Apply filters and create custom export
                    if st.button("📊 Create Filtered Export"):
                        filtered_data = combined_data.copy()
                        
                        # Apply date filters
                        if start_date_filter:
                            filtered_data = filtered_data[pd.to_datetime(filtered_data['Date']).dt.date >= start_date_filter]
                        
                        if end_date_filter:
                            filtered_data = filtered_data[pd.to_datetime(filtered_data['Date']).dt.date <= end_date_filter]
                        
                        # Apply column selection
                        if selected_columns:
                            filtered_data = filtered_data[selected_columns]
                        
                        if len(filtered_data) > 0:
                            # Create filtered filename
                            filter_suffix = ""
                            if start_date_filter or end_date_filter:
                                start_str = start_date_filter.strftime("%Y%m%d") if start_date_filter else "start"
                                end_str = end_date_filter.strftime("%Y%m%d") if end_date_filter else "end"
                                filter_suffix += f"_{start_str}_to_{end_str}"
                            
                            if len(selected_columns) < len(all_columns):
                                filter_suffix += f"_{len(selected_columns)}cols"
                            
                            filtered_filename = custom_filename.replace('.csv', f'{filter_suffix}_filtered.csv')
                            filtered_csv = filtered_data.to_csv(index=False)
                            
                            st.success(f"✅ Filtered dataset ready: {len(filtered_data):,} rows, {len(filtered_data.columns)} columns")
                            
                            st.download_button(
                                "📥 Download Filtered Data",
                                data=filtered_csv,
                                file_name=filtered_filename,
                                mime="text/csv",
                                help="Download the filtered dataset"
                            )
                        else:
                            st.warning("⚠️ No data matches the selected filters")
                
                # Save to session state option
                st.subheader("💾 Session Management")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("💾 Save to Session", help="Keep this data available for further analysis in this session"):
                        st.session_state['saved_combined_data'] = combined_data
                        st.session_state['saved_timeframe'] = target_timeframe
                        st.session_state['saved_timestamp'] = datetime.now().isoformat()
                        st.success("✅ Data saved to session memory")
                
                with col2:
                    if st.session_state.get('saved_combined_data') is not None:
                        saved_time = st.session_state.get('saved_timestamp', 'Unknown')
                        st.info(f"📊 Session data available from: {saved_time}")
                        
                        if st.button("🔄 Use Saved Data"):
                            st.info("🔗 **Ready**: Use the saved data for ATR analysis")
                
                # Integration note
                st.info("🔗 **Ready for ATR Analysis**: This processed dataset can now be used in your ATR analysis system")
    
    # Show session data if available
    if st.session_state.get('multifile_ready', False):
        st.subheader("✅ Processed Data Available")
        data = st.session_state['multifile_data']
        timeframe = st.session_state['multifile_timeframe']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total Rows", f"{len(data):,}")
        with col2:
            st.metric("📅 Date Range", f"{data['Date'].min()} to {data['Date'].max()}")
        with col3:
            st.metric("⏱️ Timeframe", timeframe)
        
        if st.button("🔄 Use This Data for ATR Analysis"):
            st.info("🔗 **Integration**: Pass this data to your main ATR analysis function")
            # Here you would integrate with your main ATR analysis system

if __name__ == "__main__":
    create_enhanced_multifile_interface()