import os
import pandas as pd
import numpy as np
import logging
from autoeda_plus.core.eda_pipeline import run_anomaly_pipeline, setup_logging

# Setup high-level logging for audit
logger = setup_logging()

def audit_functionality():
    logger.info("🕵️ Starting Functional Audit of AutoEDA++...")
    
    # 1. Create a "Fragmented" Parquet Dataset (Multi-file loading test)
    # File A: 200 rows, File B: 200 rows. Pathological numeric IDs.
    np.random.seed(42)
    df_a = pd.DataFrame(np.random.randn(200, 3), columns=['s1', 's2', 's3'])
    df_a['id'] = range(1, 201)
    df_a['target'] = (df_a['s1'] > 1.5).astype(int)
    
    df_b = pd.DataFrame(np.random.randn(200, 3), columns=['s1', 's2', 's3'])
    df_b['id'] = range(201, 401)
    df_b['target'] = (df_b['s1'] > 1.5).astype(int)
    
    df_a.to_parquet("audit_train_1.parquet", index=False)
    df_b.to_parquet("audit_train_2.parquet", index=False)
    
    # 2. Create a "Test" file with an "UNKNOWN" category in a special column
    df_test = pd.DataFrame(np.random.randn(50, 3), columns=['s1', 's2', 's3'])
    df_test['id'] = range(1001, 1051)
    df_test['cat_feature'] = "NewCategory" # Training has no 'cat_feature', so this tests 'unseen' logic
    # Wait, if training has no 'cat_feature', it will be dropped from Test during transform.
    # Let's add 'cat_feature' to Train too but with different values.
    
    df_a['cat_feature'] = "Known"
    df_b['cat_feature'] = "Known"
    df_a.to_parquet("audit_train_1.parquet", index=False)
    df_b.to_parquet("audit_train_2.parquet", index=False)
    
    df_test.to_parquet("audit_test.parquet", index=False)
    
    logger.info("✅ Mock datasets generated (including 'NewCategory' for Unknown test).")
    
    # 3. Run Pipeline
    try:
        results = run_anomaly_pipeline(
            file_paths=["audit_train_1.parquet", "audit_train_2.parquet"],
            target_col="target",
            test_file="audit_test.parquet",
            use_smote=True,
            tune_models=False,
            output_path="output/Audit_Report.ipynb"
        )
        logger.info("✅ Pipeline execution successful.")
        
        # 4. Verify Artifacts
        out_dir = results['artifact_dir']
        files = os.listdir(out_dir)
        required = ['feature_pipeline.joblib', 'best_model.joblib', 'submission.csv', 'Audit_Report.ipynb']
        
        for r in required:
            if r in files:
                logger.info(f"✅ Found artifact: {r}")
            else:
                logger.error(f"❌ Missing artifact: {r}")
                
        # 5. Verify Submission Content
        sub = pd.read_csv(os.path.join(out_dir, "submission.csv"))
        if sub['id'].iloc[0] == 1001 and len(sub) == 50:
            logger.info("✅ Submission IDs and row count verified.")
        else:
            logger.error(f"❌ Submission mismatch: Head={sub['id'].iloc[0]}, Len={len(sub)}")
            
    except Exception as e:
        logger.error(f"❌ Audit Failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        for f in ["audit_train_1.parquet", "audit_train_2.parquet", "audit_test.parquet"]:
             if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    audit_functionality()
