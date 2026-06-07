import os
import shutil
import tempfile
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

from src.dataset_manager import DatasetManager
from src.metrics_extractor import MetricsExtractor
from src.git_miner import GitMiner
from src.model_trainer import ModelTrainer

# Set up page configurations
st.set_page_config(
    page_title="AI-Powered Code Quality & Defect Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    /* Premium dark-mode gradient header */
    .header-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .header-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Sleek card container */
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s, border-color 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #3b82f6;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #3b82f6;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Code area styling */
    .code-container {
        border-left: 4px solid #3b82f6;
        background-color: #0f172a;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        font-family: 'Courier New', Courier, monospace;
    }
    
    /* Advice box */
    .advice-box {
        background-color: #1e1b4b;
        border-left: 4px solid #818cf8;
        padding: 1rem 1.5rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    .advice-title {
        font-weight: 700;
        color: #c7d2fe;
    }
    .advice-text {
        font-size: 0.95rem;
        color: #e0e7ff;
        margin-top: 0.25rem;
    }
    
    /* General spacing */
    .section-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        color: #f8fafc;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "promise_model" not in st.session_state:
    st.session_state["promise_model"] = None
if "promise_features" not in st.session_state:
    st.session_state["promise_features"] = None
if "git_model" not in st.session_state:
    st.session_state["git_model"] = None
if "git_features" not in st.session_state:
    st.session_state["git_features"] = None
if "git_data" not in st.session_state:
    st.session_state["git_data"] = None
if "repo_path" not in st.session_state:
    st.session_state["repo_path"] = None

# Header Banner
st.markdown("""
<div class="header-container">
    <h1 class="header-title">AI-Powered Code Quality & Defect Analyzer</h1>
    <p class="header-subtitle">Analyze, visualize, and predict code defects using Git repository mining, static analysis metrics, and machine learning models trained on PROMISE research datasets.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://img.icons8.com/nolan/128/artificial-intelligence.png", width=80)
st.sidebar.markdown("### Configuration Panel")

# Tabs definition
tab1, tab2, tab3 = st.tabs([
    "📊 PROMISE Benchmark Explorer", 
    "📁 Git Repository Analyzer", 
    "🔍 Single File Inspector"
])

# ----------------------------------------------------
# TAB 1: PROMISE Benchmark Explorer
# ----------------------------------------------------
with tab1:
    st.markdown("### PROMISE Defect Dataset Analysis & Predictor Training")
    st.write("Train classification models on standard software engineering research datasets containing object-oriented metrics (LOC, coupling, inheritance depth) and defect labels.")
    
    # Dataset Selector
    dm = DatasetManager()
    dataset_key = st.selectbox(
        "Select Research Dataset",
        options=list(DatasetManager.DATASETS.keys()),
        format_func=lambda k: DatasetManager.DATASETS[k]["name"]
    )
    
    try:
        raw_df = dm.load_dataset(dataset_key)
        df, feature_cols, target_col, metadata_cols = dm.preprocess_dataset(raw_df)
        
        # Summary metrics
        total_files = len(df)
        buggy_files = df[target_col].sum()
        bug_ratio = buggy_files / total_files
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Analysed Classes</div>
                <div class="metric-value">{total_files}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #ef4444;">{buggy_files}</div>
                <div class="metric-label">Defective (Buggy) Classes</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #10b981;">{bug_ratio:.1%}</div>
                <div class="metric-label">Defect Density Rate</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<div class='section-title'>Metric Distributions & Relationships</div>", unsafe_allow_html=True)
        
        # Grid of charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Bug vs Clean Pie
            fig_pie = px.pie(
                names=["Clean File", "Defective File"],
                values=[total_files - buggy_files, buggy_files],
                color_discrete_sequence=["#10b981", "#ef4444"],
                title="Class Defect Distribution",
                hole=0.4
            )
            fig_pie.update_layout(template="plotly_dark")
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with chart_col2:
            # Scatter Plot: LOC vs CBO (Coupling)
            x_metric = st.selectbox("X-Axis Metric", options=feature_cols, index=feature_cols.index('loc') if 'loc' in feature_cols else 0)
            y_metric = st.selectbox("Y-Axis Metric", options=feature_cols, index=feature_cols.index('cbo') if 'cbo' in feature_cols else 1)
            
            fig_scatter = px.scatter(
                df, x=x_metric, y=y_metric, 
                color=df['is_buggy'].map({1: 'Defective', 0: 'Clean'}),
                color_discrete_map={'Defective': '#ef4444', 'Clean': '#10b981'},
                title=f"Relationship: {x_metric.upper()} vs {y_metric.upper()}",
                hover_data=['name'] if 'name' in df.columns else []
            )
            fig_scatter.update_layout(template="plotly_dark")
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        # Correlation Heatmap
        st.markdown("<div class='section-title'>OO Metrics Correlation Matrix</div>", unsafe_allow_html=True)
        selected_heatmap_metrics = ['loc', 'wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom', 'is_buggy']
        heatmap_metrics = [m for m in selected_heatmap_metrics if m in df.columns]
        
        corr_matrix = df[heatmap_metrics].corr()
        fig_heat = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            aspect="auto",
            title="Correlation Map (CK Metrics & Bugs)"
        )
        fig_heat.update_layout(template="plotly_dark")
        st.plotly_chart(fig_heat, use_container_width=True)
        
        # Model Training UI
        st.markdown("<div class='section-title'>Train Code Quality Classifier</div>", unsafe_allow_html=True)
        train_box_col1, train_box_col2 = st.columns([1, 2])
        
        with train_box_col1:
            st.write("Configure model parameters:")
            model_type = st.radio(
                "Select Machine Learning Model",
                options=["random_forest", "logistic_regression", "xgboost"],
                format_func=lambda x: x.replace('_', ' ').title()
            )
            
            # Hyperparameters based on selection
            model_params = {}
            if model_type == "random_forest":
                model_params["n_estimators"] = st.slider("Number of Trees", min_value=10, max_value=300, value=100, step=10)
                model_params["max_depth"] = st.slider("Max Tree Depth", min_value=3, max_value=30, value=10)
            elif model_type == "xgboost":
                model_params["n_estimators"] = st.slider("Number of Boosting Rounds", min_value=10, max_value=300, value=100, step=10)
                model_params["learning_rate"] = st.slider("Learning Rate", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
                model_params["max_depth"] = st.slider("Max Tree Depth", min_value=2, max_value=15, value=5)
            elif model_type == "logistic_regression":
                model_params["C"] = st.select_slider("Inverse Regularization Strength (C)", options=[0.01, 0.1, 1.0, 10.0, 100.0], value=1.0)
                
            test_pct = st.slider("Test Split Proportion (%)", min_value=10, max_value=40, value=20, step=5) / 100.0
            
            train_btn = st.button("Train Model Now", type="primary")
            
        with train_box_col2:
            if train_btn:
                with st.spinner("Executing model training pipeline..."):
                    # Train model
                    trainer = ModelTrainer(model_type=model_type, **model_params)
                    metrics, eval_plots = trainer.train(df, feature_cols, target_col, test_size=test_pct)
                    
                    # Store model in session state for tab 3
                    st.session_state["promise_model"] = trainer
                    st.session_state["promise_features"] = feature_cols
                    
                    # Display metrics
                    metric_grid1, metric_grid2, metric_grid3, metric_grid4 = st.columns(4)
                    metric_grid1.metric("ROC-AUC Score", f"{metrics['roc_auc']:.3f}")
                    metric_grid2.metric("F1-Score", f"{metrics['f1_score']:.3f}")
                    metric_grid3.metric("Precision", f"{metrics['precision']:.3f}")
                    metric_grid4.metric("Recall", f"{metrics['recall']:.3f}")
                    
                    # Graph metrics
                    eval_col1, eval_col2 = st.columns(2)
                    with eval_col1:
                        # Confusion Matrix
                        cm = eval_plots["confusion_matrix"]
                        fig_cm = px.imshow(
                            cm,
                            x=["Clean", "Defective"],
                            y=["Clean", "Defective"],
                            color_continuous_scale="Blues",
                            text_auto=True,
                            title="Confusion Matrix"
                        )
                        fig_cm.update_layout(template="plotly_dark")
                        st.plotly_chart(fig_cm, use_container_width=True)
                    
                    with eval_col2:
                        # ROC Curve
                        roc_curve_data = eval_plots["roc_curve"]
                        fig_roc = go.Figure()
                        fig_roc.add_trace(go.Scatter(
                            x=roc_curve_data["fpr"],
                            y=roc_curve_data["tpr"],
                            mode='lines',
                            name=f'ROC Curve (AUC: {metrics["roc_auc"]:.2f})',
                            line=dict(color='#3b82f6', width=3)
                        ))
                        fig_roc.add_trace(go.Scatter(
                            x=[0, 1], y=[0, 1],
                            mode='lines',
                            name='Random Baseline',
                            line=dict(color='grey', dash='dash')
                        ))
                        fig_roc.update_layout(
                            title="ROC Curve",
                            xaxis_title="False Positive Rate",
                            yaxis_title="True Positive Rate",
                            template="plotly_dark"
                        )
                        st.plotly_chart(fig_roc, use_container_width=True)
                        
                    # Feature Importance
                    st.markdown("#### Feature Importances")
                    imp_df = trainer.get_feature_importance()
                    fig_imp = px.bar(
                        imp_df,
                        x='importance',
                        y='feature',
                        orientation='h',
                        color='importance',
                        color_continuous_scale='Viridis',
                        title='Metric Predictive Weight'
                    )
                    fig_imp.update_layout(template="plotly_dark", yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_imp, use_container_width=True)
            else:
                if st.session_state["promise_model"] is not None:
                    st.info(f"Loaded trained model: {st.session_state['promise_model'].model_type.upper()}. You can use it in Tab 3 or retrain above.")
                else:
                    st.info("Click 'Train Model Now' to fit a machine learning model on this dataset.")
                    
    except Exception as err:
        st.error(f"Error loading benchmark dataset: {err}")

# ----------------------------------------------------
# TAB 2: Git Repository Analyzer
# ----------------------------------------------------
with tab2:
    st.markdown("### End-to-End Git Repository Mining & Defect Analysis")
    st.write("Mine any local or remote Git repository. Commits are scanned for bug fixes and lines added/deleted (churn) while static metrics (LOC, complexity) are extracted from source files.")
    
    # Presets & Input
    git_presets = {
        "Custom Repo / Local Folder": "",
        "Pallets Flask (Sleek Python Web Framework)": "https://github.com/pallets/flask.git",
        "Requests (Famous Python HTTP Library)": "https://github.com/psf/requests.git"
    }
    
    repo_selection = st.selectbox("Select Git Repository Preset", options=list(git_presets.keys()))
    preset_url = git_presets[repo_selection]
    
    col_inp1, col_inp2 = st.columns([2, 1])
    with col_inp1:
        repo_url = st.text_input(
            "Git Repository URL (Remote) or Local System Path", 
            value=preset_url if preset_url else "d:/code"
        )
    with col_inp2:
        max_commits = st.number_input("Max Commits to Mine", min_value=10, max_value=2000, value=200, step=50)
        
    mine_btn = st.button("Start Git & Code Analysis", type="primary")
    
    # Analyze block
    if mine_btn:
        with st.spinner("Cloning, mining Git logs, and performing static metrics analysis..."):
            temp_dir = None
            is_temp = False
            
            try:
                # Determine repository path
                target_path = repo_url
                
                # Check if it's a URL
                if repo_url.startswith(("http://", "https://", "git@")):
                    is_temp = True
                    temp_dir = os.path.join(tempfile.gettempdir(), "git_analyzer_repo")
                    if os.path.exists(temp_dir):
                        try:
                            shutil.rmtree(temp_dir)
                        except Exception:
                            # If locked, append a random number
                            import random
                            temp_dir = f"{temp_dir}_{random.randint(1000, 9999)}"
                    
                    st.write(f"Cloning remote repository to temporary workspace...")
                    miner = GitMiner.clone_repo(repo_url, temp_dir)
                    target_path = temp_dir
                else:
                    miner = GitMiner(target_path)
                    
                st.write("Opening repository and walking through git logs...")
                git_metrics_df = miner.create_git_dataset(max_commits=max_commits)
                
                if git_metrics_df.empty:
                    st.error("No commit history found or repository couldn't be mined (could be empty or lack .py/.java files).")
                else:
                    st.write("Running static code quality analysis on Python & Java files...")
                    static_metrics = MetricsExtractor.analyze_project(target_path)
                    
                    if not static_metrics:
                        st.warning("No Python (.py) or Java (.java) source files found in the current tree of the repository.")
                        # Create empty dataset with git metrics only
                        joined_df = git_metrics_df
                        # Add fallback columns for ML
                        for col in ['loc', 'wmc', 'num_methods', 'avg_cc', 'max_cc', 'cbo', 'dit', 'noc']:
                            joined_df[col] = 0.0
                    else:
                        static_df = pd.DataFrame.from_dict(static_metrics, orient='index').reset_index()
                        static_df = static_df.rename(columns={'index': 'filepath'})
                        
                        # Merge git metrics with static metrics
                        joined_df = pd.merge(git_metrics_df, static_df, on='filepath', how='inner')
                        
                    if joined_df.empty:
                        st.error("Merge between git history files and current workspace files yielded 0 matches. Files might have been deleted/moved.")
                    else:
                        st.session_state["git_data"] = joined_df
                        st.session_state["repo_path"] = target_path
                        st.success(f"Successfully analyzed {len(joined_df)} files!")
                        
            except Exception as e:
                st.error(f"Failed to analyze repository: {e}")
                import traceback
                st.error(traceback.format_exc())
            finally:
                pass # Keep temp files or clean up? Keep for analysis access, clean later if needed

    # Display results if present in session state
    if st.session_state["git_data"] is not None:
        df = st.session_state["git_data"]
        
        st.markdown("<div class='section-title'>Mined Repository Dataset</div>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)
        
        # Grid of charts
        git_col1, git_col2 = st.columns(2)
        
        with git_col1:
            st.markdown("#### Top 10 Most Modified Files (Churn)")
            top_churn = df.nlargest(10, 'code_churn')
            fig_churn = px.bar(
                top_churn, x='code_churn', y='filepath', 
                orientation='h', color='commit_frequency',
                color_continuous_scale='Viridis',
                title="Top files by Code Churn"
            )
            fig_churn.update_layout(template="plotly_dark", yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_churn, use_container_width=True)
            
        with git_col2:
            st.markdown("#### Code Hotspots: Churn vs. Complexity")
            # Map sizing
            fig_hot = px.scatter(
                df, x='code_churn', y='max_cc',
                size='commit_frequency', color='bug_fix_count',
                color_continuous_scale='Reds',
                hover_name='filepath',
                title="Sloc Churn vs Cyclomatic Complexity (Sized by commits)"
            )
            fig_hot.update_layout(template="plotly_dark")
            st.plotly_chart(fig_hot, use_container_width=True)
            
        # Model training on this repo data
        st.markdown("<div class='section-title'>Train Git & Static Code Defect Predictor</div>", unsafe_allow_html=True)
        
        # Features to train on
        ml_features = ['commit_frequency', 'code_churn', 'num_authors', 'loc', 'wmc', 'num_methods', 'avg_cc', 'max_cc']
        # Intersect features to make sure they exist
        ml_features = [f for f in ml_features if f in df.columns]
        
        # Check if we have positive classes (buggy)
        buggy_count = df['is_buggy'].sum()
        clean_count = len(df) - buggy_count
        
        st.write(f"Repository details: **{buggy_count}** defective files, **{clean_count}** clean files.")
        
        if buggy_count < 2:
            st.warning("Defective file count is too low (< 2) to build a robust classifier on this repository's own git history. We recommend using a model pre-trained on the PROMISE benchmark (Tab 1) instead.")
        
        # Let user run training
        col_tr1, col_tr2 = st.columns([1, 2])
        with col_tr1:
            model_type_git = st.radio("Model Type", ["xgboost", "random_forest"], key="git_model_sel")
            run_git_tr = st.button("Train Repos-Specific Model", disabled=(buggy_count < 2))
        
        with col_tr2:
            if run_git_tr and buggy_count >= 2:
                with st.spinner("Fitting model..."):
                    trainer_git = ModelTrainer(model_type=model_type_git)
                    metrics_git, eval_plots_git = trainer_git.train(df, ml_features, 'is_buggy', test_size=0.25)
                    st.session_state["git_model"] = trainer_git
                    st.session_state["git_features"] = ml_features
                    
                    st.success("Trained repo classifier successfully!")
                    
                    mg1, mg2, mg3 = st.columns(3)
                    mg1.metric("ROC-AUC Score", f"{metrics_git['roc_auc']:.2f}")
                    mg2.metric("Precision", f"{metrics_git['precision']:.2f}")
                    mg3.metric("Recall", f"{metrics_git['recall']:.2f}")
                    
        # Apply model to predict defect scores
        st.markdown("<div class='section-title'>Defect Hotspot Predictions Map</div>", unsafe_allow_html=True)
        
        # Choose predictor
        predictor = None
        features_to_use = []
        predictor_source = ""
        
        if st.session_state["git_model"] is not None:
            predictor = st.session_state["git_model"]
            features_to_use = st.session_state["git_features"]
            predictor_source = "Repository Trained Model"
        elif st.session_state["promise_model"] is not None:
            # Check if features are present
            promise_feats = st.session_state["promise_features"]
            # Map repo features to promise features
            # PROMISE features: wmc, dit, noc, cbo, rfc, lcom, loc, max_cc, avg_cc
            available_promise_feats = [f for f in promise_feats if f in df.columns]
            if len(available_promise_feats) >= 3:
                predictor = st.session_state["promise_model"]
                features_to_use = promise_feats
                predictor_source = "PROMISE Pre-Trained Model"
                
                # Fill missing columns in df with 0
                for f in promise_feats:
                    if f not in df.columns:
                        df[f] = 0.0
            else:
                st.info("PROMISE pre-trained model needs CK features which are not fully mapped here.")
        
        if predictor is not None:
            st.info(f"Using predictor: **{predictor_source}** to evaluate file defect probability.")
            
            # Predict probabilities
            df['defect_probability'] = predictor.predict_proba(df)
            
            # Sort by risk
            df_risk = df.sort_values(by='defect_probability', ascending=False)
            
            # Risk categorisation
            def get_risk_level(prob):
                if prob >= 0.7: return "🔴 High Risk"
                elif prob >= 0.4: return "🟡 Medium Risk"
                else: return "🟢 Low Risk"
            df_risk['Risk Level'] = df_risk['defect_probability'].apply(get_risk_level)
            
            # Bubble chart
            fig_risk_bubble = px.scatter(
                df_risk, x='code_churn', y='max_cc',
                size='loc', color='defect_probability',
                color_continuous_scale='Jet',
                hover_name='filepath',
                hover_data=['Risk Level', 'commit_frequency'],
                title="Visual Defect Hotspots (Sized by LOC, Color by Bug Probability)"
            )
            fig_risk_bubble.update_layout(template="plotly_dark")
            st.plotly_chart(fig_risk_bubble, use_container_width=True)
            
            # Show high risk hotspots table
            st.markdown("#### High & Medium Defect Risk Hotspots (Action Required)")
            hotspots_df = df_risk[df_risk['defect_probability'] >= 0.4][['filepath', 'Risk Level', 'defect_probability', 'loc', 'max_cc', 'code_churn', 'commit_frequency']]
            
            if hotspots_df.empty:
                st.success("No files flagged as High or Medium risk! Your code appears healthy.")
            else:
                st.dataframe(hotspots_df, use_container_width=True)
        else:
            st.warning("To generate defect hotspot predictions, first train a model in Tab 1 (PROMISE dataset) or Tab 2 (Git Repository Specific).")

# ----------------------------------------------------
# TAB 3: Single File Inspector
# ----------------------------------------------------
with tab3:
    st.markdown("### Real-Time Single File Inspector & Refactoring assistant")
    st.write("Upload, select, or paste source code (Python/Java) to run real-time static code quality analysis and predict its defect likelihood.")
    
    inspect_col1, inspect_col2 = st.columns([1, 1])
    
    with inspect_col1:
        st.markdown("#### Code Input")
        
        # Option to paste code or load from analyzed repo
        input_source = st.radio("Code Source", ["Paste Code Snippet", "Select Mined File from Tab 2"])
        
        code_text = ""
        lang = "Python"
        
        if input_source == "Select Mined File from Tab 2":
            if st.session_state["git_data"] is not None:
                files_list = st.session_state["git_data"]['filepath'].tolist()
                selected_file = st.selectbox("Select File to Inspect", options=files_list)
                
                # Resolve file path from stored repo workspace
                base_path = st.session_state.get("repo_path") or "."
                filepath_full = os.path.join(base_path, selected_file)
                    
                if os.path.exists(filepath_full):
                    try:
                        with open(filepath_full, 'r', encoding='utf-8', errors='ignore') as f:
                            code_text = f.read()
                        lang = "Java" if selected_file.endswith('.java') else "Python"
                    except Exception as err:
                        st.error(f"Failed to read file contents: {err}")
                else:
                    st.error(f"File path does not exist locally: {filepath_full}")
            else:
                st.warning("Please analyze a Git Repository in Tab 2 first.")
                input_source = "Paste Code Snippet"
                
        if input_source == "Paste Code Snippet":
            lang = st.selectbox("Programming Language", ["Python", "Java"])
            default_code = ""
            if lang == "Python":
                default_code = """def calculate_factorials(numbers):
    # This is a sample code snippet with high cyclomatic complexity
    results = []
    for num in numbers:
        if num < 0:
            results.append(None)
        elif num == 0 or num == 1:
            results.append(1)
        else:
            fact = 1
            for i in range(2, num + 1):
                if i % 2 == 0:
                    fact *= i
                else:
                    fact += i
            results.append(fact)
    return results
"""
            else:
                default_code = """public class Utility {
    public int process(int val, String type) {
        // High complexity java method
        int result = 0;
        if (type.equals("add")) {
            for (int i = 0; i < val; i++) {
                if (i % 2 == 0 && val > 10) {
                    result += i;
                } else if (val == 5) {
                    result += 5;
                } else {
                    result++;
                }
            }
        } else if (type.equals("mult")) {
            result = 1;
            while (val > 1) {
                result *= val;
                val--;
            }
        } else {
            result = -1;
        }
        return result;
    }
}
"""
            code_text = st.text_area("Paste code here:", value=default_code, height=350)
            
        analyze_single_btn = st.button("Inspect Code Quality", type="primary")
        
    with inspect_col2:
        st.markdown("#### Code Quality Report")
        
        if analyze_single_btn and code_text:
            with st.spinner("Extracting static metrics..."):
                # Save code snippet to a temp file to analyze
                with tempfile.NamedTemporaryFile(suffix=".py" if lang == "Python" else ".java", delete=False, mode='w', encoding='utf-8') as tf:
                    tf.write(code_text)
                    temp_filepath = tf.name
                    
                try:
                    # Run metric extractor on temp file
                    metrics = MetricsExtractor.analyze_file(temp_filepath)
                    
                    if not metrics or "error" in metrics:
                        st.error(f"Error parsing file: {metrics.get('error', 'unknown error')}")
                    else:
                        # KPI cards for metrics
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Lines of Code (LOC)", metrics.get("loc", 0))
                        m2.metric("Weighted CC (WMC)", metrics.get("wmc", 0))
                        m3.metric("Avg CC / Method", f"{metrics.get('avg_cc', 0):.2f}")
                        
                        m4, m5 = st.columns(2)
                        m4.metric("Number of Methods", metrics.get("num_methods", 0))
                        m5.metric("Class Coupling (CBO)", metrics.get("cbo", 0))
                        
                        # Defect likelihood using ML predictor
                        st.markdown("<div class='section-title'>Defect Likelihood Prediction</div>", unsafe_allow_html=True)
                        
                        predictor = None
                        features_list = []
                        
                        if st.session_state["promise_model"] is not None:
                            predictor = st.session_state["promise_model"]
                            features_list = st.session_state["promise_features"]
                            st.write("Predicting using **PROMISE Pre-Trained Model**")
                        elif st.session_state["git_model"] is not None:
                            predictor = st.session_state["git_model"]
                            features_list = st.session_state["git_features"]
                            st.write("Predicting using **Git Repository Trained Model**")
                            
                        if predictor is not None:
                            # Build a single-row DataFrame matching the model's features
                            single_data = {}
                            for feat in features_list:
                                # Map values from extracted metrics
                                val = 0.0
                                if feat == 'loc': val = metrics.get('loc', 0.0)
                                elif feat == 'wmc': val = metrics.get('wmc', 0.0)
                                elif feat == 'max_cc': val = metrics.get('max_cc', 0.0)
                                elif feat == 'avg_cc': val = metrics.get('avg_cc', 0.0)
                                elif feat == 'num_methods': val = metrics.get('num_methods', 0.0)
                                elif feat == 'cbo': val = metrics.get('cbo', 0.0)
                                elif feat == 'dit': val = metrics.get('dit', 1.0)
                                elif feat == 'noc': val = metrics.get('noc', 0.0)
                                # Fill git features with dataset averages or 0 if single file has no git context
                                single_data[feat] = float(val)
                                
                            single_df = pd.DataFrame([single_data])
                            prob = predictor.predict_proba(single_df)[0]
                            
                            # Custom visual gauge or bar
                            color_bar = "#10b981" if prob < 0.4 else ("#eab308" if prob < 0.7 else "#ef4444")
                            st.markdown(f"""
                            <div style="background-color: #1e293b; padding: 1rem; border-radius: 8px; border: 1px solid #334155;">
                                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                                    <span style="font-weight: 600; color: #94a3b8;">Defect Risk:</span>
                                    <span style="font-weight: 700; color: {color_bar};">{prob:.1%}</span>
                                </div>
                                <div style="background-color: #475569; border-radius: 4px; height: 10px; width: 100%;">
                                    <div style="background-color: {color_bar}; height: 10px; border-radius: 4px; width: {prob*100}%;"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.warning("No model trained yet. Defect prediction is unavailable. (Train one in Tab 1 or Tab 2 first)")
                            
                        # Refactoring recommendations
                        st.markdown("<div class='section-title'>Refactoring Recommendations</div>", unsafe_allow_html=True)
                        
                        has_advice = False
                        
                        if metrics.get("avg_cc", 0) > 4:
                            has_advice = True
                            st.markdown("""
                            <div class="advice-box" style="border-left-color: #f59e0b;">
                                <div class="advice-title">⚠️ High Method Complexity</div>
                                <div class="advice-text">The average cyclomatic complexity of your methods is high. Consider breaking down complex blocks, extracting nested loops/conditions into helper methods, and simplifying conditions.</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        if metrics.get("loc", 0) > 200:
                            has_advice = True
                            st.markdown("""
                            <div class="advice-box" style="border-left-color: #ef4444;">
                                <div class="advice-title">⚠️ Large File (LOC > 200)</div>
                                <div class="advice-text">The class contains a large number of lines. High LOC increases cognitive load. Consider partitioning the file into smaller modules following the Single Responsibility Principle.</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        if metrics.get("cbo", 0) > 8:
                            has_advice = True
                            st.markdown("""
                            <div class="advice-box" style="border-left-color: #818cf8;">
                                <div class="advice-title">⚠️ Strong Class Coupling (CBO)</div>
                                <div class="advice-text">High coupling between objects makes code fragile to change. Consider using dependency injection, interfaces, or event handlers to decouple classes.</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        if not has_advice:
                            st.success("🎉 Excellent! Your code is highly cohesive, modular, and clean. No refactoring necessary.")
                            
                except Exception as err:
                    st.error(f"Failed to analyze snippet: {err}")
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_filepath):
                        os.unlink(temp_filepath)
        else:
            st.info("Paste your code and click 'Inspect Code Quality' to run static analysis and predictions.")
